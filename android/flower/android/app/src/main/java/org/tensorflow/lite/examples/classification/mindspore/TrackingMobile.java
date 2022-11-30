package org.tensorflow.lite.examples.classification.mindspore;

import android.content.Context;
import android.content.res.AssetFileDescriptor;
import android.graphics.Bitmap;
import android.util.Log;

import com.mindspore.MSTensor;
import com.mindspore.Model;
import com.mindspore.config.CpuBindMode;
import com.mindspore.config.DeviceType;
import com.mindspore.config.MSContext;
import com.mindspore.config.ModelType;

import java.io.FileInputStream;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.FloatBuffer;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;
import java.util.HashSet;
import java.util.List;

public class TrackingMobile {
    private static final String TAG = "TrackingMobile";

    private static final String IMAGESEGMENTATIONMODEL = "resnet50.ms";
    private static final int imageSize = 256;
    //    private static final int imageSize = 257;
    private static final float IMAGE_MEAN = 127.5F;
    private static final float IMAGE_STD = 127.5F;

    public static final int[] segmentColors = new int[2];

    private Bitmap maskBitmap;
    private Bitmap resultBitmap;
    private HashSet itemsFound = new HashSet();

    private final Context mContext;

    private Model model;

    public TrackingMobile(Context context) {
        mContext = context;
        init();
    }

    private MappedByteBuffer loadModel(Context context, String modelName) {
        FileInputStream fis = null;
        AssetFileDescriptor fileDescriptor = null;

        try {
            fileDescriptor = context.getAssets().openFd(modelName);
            fis = new FileInputStream(fileDescriptor.getFileDescriptor());
            FileChannel fileChannel = fis.getChannel();
            long startOffset = fileDescriptor.getStartOffset();
            long declaredLen = fileDescriptor.getDeclaredLength();
            return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLen);
        } catch (IOException var24) {
            Log.e("MS_LITE", "Load model failed");
        } finally {
            if (fis != null) {
                try {
                    fis.close();
                } catch (IOException var23) {
                    Log.e("MS_LITE", "Close file failed");
                }
            }

            if (fileDescriptor != null) {
                try {
                    fileDescriptor.close();
                } catch (IOException var22) {
                    Log.e("MS_LITE", "Close fileDescriptor failed");
                }
            }

        }

        return null;
    }

    public void init() {
        // Load the .ms model.
        model = new Model();

        // Create and init config.
        MSContext context = new MSContext();
        if (!context.init(2, CpuBindMode.MID_CPU, false)) {
            Log.e(TAG, "Init context failed");
            return;
        }
        if (!context.addDeviceInfo(DeviceType.DT_CPU, false, 0)) {
            Log.e(TAG, "Add device info failed");
            return;
        }
        MappedByteBuffer modelBuffer = loadModel(mContext, IMAGESEGMENTATIONMODEL);
        if (modelBuffer == null) {
            Log.e(TAG, "Load model failed");
            return;
        }
        // Create the MindSpore lite session.
        boolean ret = model.build(modelBuffer, ModelType.MT_MINDIR, context);
        if (!ret) {
            Log.e(TAG, "Build model failed");
        }
    }

    public FlowerResultBean execute(Bitmap bitmap) {
        // Set input tensor values.
        List<MSTensor> inputs = model.getInputs();
        if (inputs.size() != 1) {
            Log.e(TAG, "inputs.size() != 1");
            return null;
        }
//
//        float resource_height = bitmap.getHeight();
//        float resource_weight = bitmap.getWidth();

        Bitmap scaledBitmap = BitmapUtils.scaleBitmapAndKeepRatio(bitmap, imageSize, imageSize);
        ByteBuffer contentArray = BitmapUtils.bitmapToByteBuffer(scaledBitmap, imageSize, imageSize, IMAGE_MEAN, IMAGE_STD);

        Log.e(TAG, "ByteBuffer" + contentArray.array().length);
        MSTensor inTensor = inputs.get(0);
        inTensor.setData(contentArray);

        if (inTensor.getShape() != null) {
            Log.e(TAG, "inTensor.getShape().length" + inTensor.getShape().length);
            for (int i = 0; i < inTensor.getShape().length; i++) {
                Log.e(TAG, "inTensor.getShape()" + inTensor.getShape()[i]);
            }
        }

        // Run graph to infer results.
        if (!model.predict()) {
            Log.e(TAG, "Run graph failed");
            return null;
        }

        // Get output tensor values.
        MSTensor output = model.getOutputs().get(0);
        if (output == null) {
            Log.e(TAG, "Output is null");
            return null;
        }

        if (output.getShape() != null) {
            Log.e(TAG, "output.getShape().length" + output.getShape().length);
            for (int i = 0; i < output.getShape().length; i++) {
                Log.e(TAG, "output.getShape()" + output.getShape()[i]);
            }
        }


        float[] results = output.getFloatData();
        float max = 0.0f;
        int maxIndex = 0;

        float sumExp = 0.0f;
        for (int i = 0; i < 5; ++i) {
            sumExp += Math.exp(results[i]);
        }
        for (int i = 0; i < 5; ++i) {
            if (results[i] > max) {
                max = (float) (Math.exp(results[i]) / sumExp);
                maxIndex = i;
            }
        }
        // Score for each category.
        // Converted to text information that needs to be displayed in the APP.


        return new FlowerResultBean(maxIndex, max);
    }

    private static ByteBuffer floatArrayToByteArray(float[] floats) {
        ByteBuffer buffer = ByteBuffer.allocate(4 * floats.length);
        FloatBuffer floatBuffer = buffer.asFloatBuffer();
        floatBuffer.put(floats);
        return buffer;
    }


    // Note: we must release the memory at the end, otherwise it will cause the memory leak.
    public void free() {
        model.free();
    }


    private final String formatExecutionLog() {
        StringBuilder sb = new StringBuilder();
        sb.append("Input Image Size: " + imageSize * imageSize + '\n');
        return sb.toString();
    }

}