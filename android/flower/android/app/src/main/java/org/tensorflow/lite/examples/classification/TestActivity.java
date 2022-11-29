package org.tensorflow.lite.examples.classification;

import androidx.appcompat.app.AppCompatActivity;

import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.os.Bundle;
import android.util.Log;
import android.view.View;
import android.widget.TextView;

import org.tensorflow.lite.examples.classification.mindspore.FlowerResultBean;
import org.tensorflow.lite.examples.classification.mindspore.TrackingMobile;

public class TestActivity extends AppCompatActivity {

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_test);

        TextView tv = findViewById(R.id.text);
        Bitmap bitmap = BitmapFactory.decodeResource(getResources(),R.drawable.daisy).copy(Bitmap.Config.ARGB_8888,true);
        TrackingMobile trackingMobile = new TrackingMobile(this);

        tv.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                FlowerResultBean flowerResultBean = trackingMobile.execute(bitmap);
                if (flowerResultBean != null && flowerResultBean.getLabel()!=null) {
                    Log.e("AAAAAAAAAA", flowerResultBean.getLabel());
                }
            }
        });

    }
}