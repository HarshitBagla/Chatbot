package chatbot.nphb18.chatbot;

import android.content.Context;
import android.os.CountDownTimer;
import android.support.v7.app.AppCompatActivity;
import android.os.Bundle;
import android.support.v7.widget.LinearLayoutManager;
import android.support.v7.widget.RecyclerView;
import android.view.KeyEvent;
import android.view.LayoutInflater;
import android.view.View;
import android.view.inputmethod.InputMethodManager;
import android.widget.Button;
import android.widget.EditText;
import android.widget.LinearLayout;
import android.widget.ScrollView;
import android.widget.TextView;

import java.util.ArrayList;

public class MainActivity extends AppCompatActivity {

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        final EditText entry = findViewById(R.id.text_entry);
        final ScrollView container = findViewById(R.id.conversation_container);
        final LinearLayout layout = findViewById(R.id.conversation_log);

        entry.setOnEditorActionListener(new TextView.OnEditorActionListener() {
            @Override
            public boolean onEditorAction(TextView v, int actionId, KeyEvent event) {
                return true;
            }
        });
        Button enter = findViewById(R.id.text_entry_button);
        enter.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                final String text = entry.getText().toString();
                View inputView = getLayoutInflater().inflate(R.layout.message_sent, layout, false);
                TextView sentText = inputView.findViewById(R.id.sent_message);
                sentText.setText(text);
                layout.addView(inputView);
                container.fullScroll(ScrollView.FOCUS_DOWN);
                entry.setText("");

                new CountDownTimer(500, 500) {
                    public void onTick(long millisUntilFinished) {
                    }

                    public void onFinish() {
                        String response = getResponse(text);
                        View responseView = getLayoutInflater().inflate(R.layout.message_received, layout, false);
                        TextView responseText = responseView.findViewById(R.id.response_message);
                        responseText.setText(response);
                        layout.addView(responseView);
                        container.fullScroll(ScrollView.FOCUS_DOWN);
                    }
                }.start();
            }
        });
    }

    private String getResponse(String input) {
        return input + "Jai Mata Di";
    }
}
