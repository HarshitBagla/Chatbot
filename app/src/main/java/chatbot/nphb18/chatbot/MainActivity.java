package chatbot.nphb18.chatbot;

import android.support.v7.app.AppCompatActivity;
import android.os.Bundle;
import android.view.KeyEvent;
import android.view.View;
import android.widget.Button;
import android.widget.EditText;
import android.widget.LinearLayout;
import android.widget.ScrollView;
import android.widget.TextView;

import com.android.volley.Request;
import com.android.volley.RequestQueue;
import com.android.volley.Response;
import com.android.volley.RetryPolicy;
import com.android.volley.VolleyError;
import com.android.volley.toolbox.StringRequest;
import com.android.volley.toolbox.Volley;

import java.util.ArrayList;

public class MainActivity extends AppCompatActivity {

    private static final String BASE_URL = "http://10.0.2.2:5000/";

    RequestQueue queue;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        queue = Volley.newRequestQueue(this);

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

                View responseView = getLayoutInflater().inflate(R.layout.message_received, layout, false);
                TextView responseText = responseView.findViewById(R.id.response_message);
                getResponse(text, responseText);
                layout.addView(responseView);
                container.fullScroll(ScrollView.FOCUS_DOWN);
            }
        });
    }

    private void getResponse(String input, TextView tv) {
        GetResponseTask task = new GetResponseTask(this, tv);
        task.execute(input);
    }
}
