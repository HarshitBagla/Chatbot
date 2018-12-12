package chatbot.nphb18.chatbot;

import android.content.Context;
import android.os.AsyncTask;
import android.widget.TextView;

import com.android.volley.DefaultRetryPolicy;
import com.android.volley.Request;
import com.android.volley.RequestQueue;
import com.android.volley.Response;
import com.android.volley.VolleyError;
import com.android.volley.toolbox.StringRequest;
import com.android.volley.toolbox.Volley;

public class GetResponseTask extends AsyncTask<String, Void, Void> {

    private Context context;
    private VolleyCallback callback;

    GetResponseTask(final Context context, final TextView tv) {
        this.context = context;
        callback = new VolleyCallback() {
            @Override
            public void onSuccess(String response) {
                tv.setText(response);
            }
        };
    }

    Context getContext() {
        return this.context;
    }

    private static final String URL = "http:10.0.2.2:5000/reply/";

    protected Void doInBackground(String... inputs) {
        RequestQueue queue = Volley.newRequestQueue(getContext());
        for (String input: inputs) {
            String url_to_use = URL + input;
            StringRequest request = new StringRequest
                    (Request.Method.GET, url_to_use, new Response.Listener<String>() {
                        @Override
                        public void onResponse(String response) {
                            callback.onSuccess(response);
                        }
                    }, new Response.ErrorListener() {
                        @Override
                        public void onErrorResponse(VolleyError error) {
                            error.printStackTrace();
                        }
                    });
            request.setRetryPolicy(new DefaultRetryPolicy(10000,
                    DefaultRetryPolicy.DEFAULT_MAX_RETRIES, DefaultRetryPolicy.DEFAULT_BACKOFF_MULT));
            queue.add(request);
        }
        return null;
    }
}
