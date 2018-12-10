import numpy as np
import tensorflow as tf
import os
import pickle
import copy
import html
import time
from model import Model


class IOManager(object):

    def __init__(self):
        self.user_input = None
        self.response = None

    def is_input_set(self):
        if self.user_input is not None:
            return True
        else:
            return False

    def get_input(self):
        return self.user_input

    def set_input(self, user_in):
        self.user_input = user_in
        self.response = None

    def add_response(self, generated_reply):
        self.response = generated_reply
        self.user_input = None

    def is_response_generated(self):
        if self.response is not None:
            return True
        else:
            return False

    def get_response(self):
        return self.response


def get_paths(input_path):
    if os.path.isfile(input_path):
        # Passed a model
        model_path = input_path
        save_dir = os.path.dirname(model_path)
    elif os.path.exists(input_path):
        # Passed a checkpoint directory
        save_dir = input_path
        checkpoint = tf.train.get_checkpoint_state(save_dir)
        if checkpoint:
            model_path = checkpoint.model_checkpoint_path
        else:
            raise ValueError('Checkpoint not found in {}.'.format(save_dir))
    else:
        raise ValueError('save_dir is not a valid path.')
    return model_path, os.path.join(save_dir, 'config.pkl'), os.path.join(save_dir, 'chars_vocab.pkl')


def get_initial_state(net, sess):
    return sess.run(net.zero_state)


def forward_text(net, sess, states, vocab, prime_text=None):
    if prime_text is not None:
        for char in prime_text:
            _, states = net.forward_model(sess, states, vocab[char])
    return states


def sanitize_text(vocab, text):  # Strip out characters that are not part of the net's vocab.
    return ''.join(i for i in text if i in vocab)


def possibly_escaped_char(raw_chars):
    if raw_chars[-1] == ';':
        for i, c in enumerate(reversed(raw_chars[:-1])):
            if c == ';' or i > 8:
                return raw_chars[-1]
            elif c == '&':
                escape_seq = "".join(raw_chars[-(i + 2):])
                new_seq = html.unescape(escape_seq)
                backspace_seq = "".join(['\b'] * (len(escape_seq)-1))
                diff_length = len(escape_seq) - len(new_seq) - 1
                return backspace_seq + new_seq + "".join([' '] * diff_length) + "".join(['\b'] * diff_length)
    return raw_chars[-1]


def initialize_bot():
    save_dir = 'models/reddit'
    max_length = 500
    beam_width = 2
    model_path, config_path, vocab_path = get_paths(save_dir)
    # Arguments passed to sample.py direct us to a saved model.
    # Load the separate arguments by which that model was previously trained.
    # That's saved_args. Use those to load the model.
    with open(config_path, 'rb') as f:
        saved_args = pickle.load(f)
    # Separately load chars and vocab from the save directory.
    with open(vocab_path, 'rb') as f:
        chars, vocab = pickle.load(f)
    # Create the model from the saved arguments, in inference mode.
    print("Creating model...")
    saved_args.batch_size = beam_width
    net = Model(saved_args, True)
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    # Make tensorflow less verbose; filter out info (1+) and warnings (2+) but not errors (3).
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    with tf.Session(config=config) as sess:
        tf.global_variables_initializer().run()
        saver = tf.train.Saver(net.save_variables_list())
        # Restore the saved variables, replacing the initialized values.
        print("Restoring weights...")
        saver.restore(sess, model_path)
        states = get_initial_state(net, sess)
        get_response(net, sess, chars, vocab, max_length, beam_width, states)


def get_response(net, sess, chars, vocab, max_length, beam_width, states):
    manager = IOManager()
    while True:
        if manager.is_input_set():
            user_input = manager.get_input()
            states = forward_text(net, sess, states, vocab, sanitize_text(vocab, "> " + user_input + "\n>"))
            computer_response_generator = beam_search_generator(sess=sess, net=net,
                                                                initial_state=copy.deepcopy(states),
                                                                initial_sample=vocab[' '],
                                                                early_term_token=vocab['\n'], beam_width=beam_width,
                                                                forward_model_fn=forward_with_mask,
                                                                forbidden_token=vocab['>'])
            out_chars = []
            final_response = ""
            for i, char_token in enumerate(computer_response_generator):
                out_chars.append(chars[char_token])
                final_response += possibly_escaped_char(out_chars)
                states = forward_text(net, sess, states, vocab, chars[char_token])
                if i >= max_length:
                    break
            states = forward_text(net, sess, states, vocab, sanitize_text(vocab, "\n> "))
            manager.add_response(final_response)
        else:
            time.sleep(0.05)


def consensus_length(beam_outputs, early_term_token):
    for l in range(len(beam_outputs[0])):
        if l > 0 and beam_outputs[0][l-1] == early_term_token:
            return l-1, True
        for b in beam_outputs[1:]:
            if beam_outputs[0][l] != b[l]:
                return l, False
    return l, False


def forward_with_mask(sess, net, states, input_sample, forbidden_token):
    prob, states = net.forward_model(sess, states, input_sample)
    # Mask out the forbidden token (">") to prevent the bot from deciding the chat is over)
    prob[forbidden_token] = 0
    # Normalize probabilities so they sum to 1.
    prob = prob / sum(prob)
    return prob, states


def beam_search_generator(sess, net, initial_state, initial_sample, early_term_token, beam_width, forward_model_fn,
                          forbidden_token):
    """
    Run beam search! Yield consensus tokens sequentially, as a generator;
    return when reaching early_term_token (newline).

    Args:
        sess: tensorflow session reference
        net: tensorflow net graph (must be compatible with the forward_net function)
        initial_state: initial hidden state of the net
        initial_sample: single token (excluding any seed/priming material)
            to start the generation
        early_term_token: stop when the beam reaches consensus on this token
            (but do not return this token).
        beam_width: how many beams to track
        forward_model_fn: function to forward the model, must be of the form:
            probability_output, beam_state =
                    forward_model_fn(sess, net, beam_state, beam_sample, forward_args)
            (Note: probability_output has to be a valid probability distribution!)
        forbidden_token: stores the forbidden token
    Returns: a generator to yield a sequence of beam-sampled tokens.
    """
    # Store state, outputs and probabilities for up to args.beam_width beams.
    # Initialize with just the one starting entry; it will branch to fill the beam
    # in the first step.
    beam_states = [initial_state]  # Stores the best activation states
    beam_outputs = [[initial_sample]]  # Stores the best generated output sequences so far.
    beam_probs = [1.]  # Stores the cumulative normalized probabilities of the beams so far.

    while True:
        # Keep a running list of the best beam branches for next step.
        # Don't actually copy any big data structures yet, just keep references
        # to existing beam state entries, and then clone them as necessary
        # at the end of the generation step.
        new_beam_indices = []
        new_beam_probs = []
        new_beam_samples = []

        # Iterate through the beam entries.
        for beam_index, beam_state in enumerate(beam_states):
            beam_prob = beam_probs[beam_index]
            beam_sample = beam_outputs[beam_index][-1]

            # Forward the model.
            prediction, beam_states[beam_index] = forward_model_fn(sess, net, beam_state, beam_sample, forbidden_token)

            # Sample best_tokens from the probability distribution.
            # Sample from the scaled probability distribution beam_width choices
            # (but not more than the number of positive probabilities in scaled_prediction).
            count = min(beam_width, sum(1 if p > 0. else 0 for p in prediction))
            best_tokens = np.random.choice(len(prediction), size=count, replace=False, p=prediction)
            for token in best_tokens:
                prob = prediction[token] * beam_prob
                if len(new_beam_indices) < beam_width:
                    # If we don't have enough new_beam_indices, we automatically qualify.
                    new_beam_indices.append(beam_index)
                    new_beam_probs.append(prob)
                    new_beam_samples.append(token)
                else:
                    # Sample a low-probability beam to possibly replace.
                    np_new_beam_probs = np.array(new_beam_probs)
                    inverse_probs = -np_new_beam_probs + max(np_new_beam_probs) + min(np_new_beam_probs)
                    inverse_probs = inverse_probs / sum(inverse_probs)
                    sampled_beam_index = np.random.choice(beam_width, p=inverse_probs)
                    if new_beam_probs[sampled_beam_index] <= prob:
                        # Replace it.
                        new_beam_indices[sampled_beam_index] = beam_index
                        new_beam_probs[sampled_beam_index] = prob
                        new_beam_samples[sampled_beam_index] = token
        # Replace the old states with the new states, first by referencing and then by copying.
        already_referenced = [False] * beam_width
        new_beam_states = []
        new_beam_outputs = []
        for i, new_index in enumerate(new_beam_indices):
            if already_referenced[new_index]:
                new_beam = copy.deepcopy(beam_states[new_index])
            else:
                new_beam = beam_states[new_index]
                already_referenced[new_index] = True
            new_beam_states.append(new_beam)
            new_beam_outputs.append(beam_outputs[new_index] + [new_beam_samples[i]])
        # Normalize the beam probabilities so they don't drop to zero
        beam_probs = new_beam_probs / sum(new_beam_probs)
        beam_states = new_beam_states
        beam_outputs = new_beam_outputs
        # Prune the agreed portions of the outputs
        # and yield the tokens on which the beam has reached consensus.
        l, early_term = consensus_length(beam_outputs, early_term_token)
        if l > 0:
            for token in beam_outputs[0][:l]:
                yield token
            beam_outputs = [output[l:] for output in beam_outputs]
        if early_term:
            return
