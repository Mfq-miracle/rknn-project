package com.example.hc.digitalgesturerecognition;

import android.Manifest;
import android.annotation.SuppressLint;
import android.annotation.TargetApi;
import android.content.Context;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.ImageFormat;
import android.graphics.Matrix;
import android.hardware.camera2.CameraAccessException;
import android.hardware.camera2.CameraCaptureSession;
import android.hardware.camera2.CameraCharacteristics;
import android.hardware.camera2.CameraDevice;
import android.hardware.camera2.CameraManager;
import android.hardware.camera2.CaptureRequest;
import android.media.Image;
import android.media.ImageReader;
import android.net.Uri;
import android.os.Build;
import android.os.Bundle;
import android.os.Environment;
import android.os.Handler;
import android.os.HandlerThread;
import android.support.annotation.NonNull;
import android.support.annotation.RequiresApi;
import android.support.v4.app.ActivityCompat;
import android.support.v7.app.AppCompatActivity;
import android.util.Log;
import android.util.SparseIntArray;
import android.view.Surface;
import android.view.SurfaceHolder;
import android.view.SurfaceView;
import android.view.View;
import android.view.Window;
import android.view.WindowManager;
import android.view.animation.Animation;
import android.view.animation.AnimationUtils;
import android.widget.ImageView;
import android.widget.TextView;
import android.widget.Toast;

import com.example.hc.digitalgesturerecognition.flashstrategy.AutoStrategy;
import com.example.hc.digitalgesturerecognition.flashstrategy.CloseStrategy;
import com.example.hc.digitalgesturerecognition.flashstrategy.FlashStrategy;
import com.example.hc.digitalgesturerecognition.flashstrategy.KeepOpenStrategy;
import com.example.hc.digitalgesturerecognition.flashstrategy.OpenStrategy;

import org.opencv.android.OpenCVLoader;
import org.opencv.android.Utils;
import org.opencv.core.Mat;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.OutputStream;
import java.nio.ByteBuffer;
import java.util.ArrayList;
import java.util.Arrays;


public class CameraActivity extends AppCompatActivity implements View.OnClickListener{
    //??????OpenCV,???????????????
    static{
        if(!OpenCVLoader.initDebug())
        {
            Log.d("OpenCV", "init failed");
        }
    }
    private static final SparseIntArray ORIENTATIONS = new SparseIntArray();
    private static final String TAG = "CameraActivity";

    ///???????????????????????????
    static {
        ORIENTATIONS.append(Surface.ROTATION_0, 90);
        ORIENTATIONS.append(Surface.ROTATION_90, 0);
        ORIENTATIONS.append(Surface.ROTATION_180, 270);
        ORIENTATIONS.append(Surface.ROTATION_270, 180);
    }
    private View mFlashLayout;
    private TextView tv_flashStatus, tv_flashClose, tv_flashOpen, tv_flashKeepOpen, tv_flashAuto;
    private ImageView iv_thumb, iv_changeCamera, iv_flash;
    private SurfaceHolder mSurfaceHolder;
    private SurfaceView mSurfaceView;
    private Handler childHandler;
    private Handler mainHandler;
    private int mCameraID;
    private ImageReader mImageReader;
    private CameraDevice mCameraDevice;//???????????????
    private CameraManager mCameraManager;
    private CameraCaptureSession mCameraCaptureSession;
    public final int FLASH_ON = 1;
    public final int FLASH_OFF = 2;
    private CaptureRequest.Builder mPreviewRequestBuilder;

    public final int BACK_CAMERA = 0; //??????????????????CameraId
    public final int FRONT_CAMERA = 1;

    private Classifier classifier;//?????????
    private static final String MODEL_FILE = "file:///android_asset/digital_gesture.pb"; //??????????????????

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        supportRequestWindowFeature(Window.FEATURE_NO_TITLE);//???????????????
        getWindow().setFlags(WindowManager.LayoutParams.FLAG_FULLSCREEN,WindowManager.LayoutParams.FLAG_FULLSCREEN);//???????????????
        setContentView(R.layout.activity_camera);

        initView();
    }

    @Override
    public void onRequestPermissionsResult(int requestCode, @NonNull String[] permissions, @NonNull int[] grantResults) {
        if(requestCode == 1){
            for (int grantResult : grantResults) {
                if(grantResult != PackageManager.PERMISSION_GRANTED){
                    Toast.makeText(this, "???????????????????????????????????????????????????", Toast.LENGTH_SHORT).show();
                    return;
                }
            }
            initCamera2();
        }
    }

    private void initView() {
        iv_flash = findViewById(R.id.iv_flash);
        iv_thumb = findViewById(R.id.iv_thumb);
        tv_flashStatus = findViewById(R.id.tv_flash_status);
        mSurfaceView = findViewById(R.id.surfaceView);
        iv_changeCamera = findViewById(R.id.iv_change);

        mFlashLayout = findViewById(R.id.layout_flash_text);
        tv_flashAuto = mFlashLayout.findViewById(R.id.tv_flash_auto);
        tv_flashClose = mFlashLayout.findViewById(R.id.tv_flash_close);
        tv_flashKeepOpen = mFlashLayout.findViewById(R.id.tv_flash_keep_open);
        tv_flashOpen =mFlashLayout.findViewById(R.id.tv_flash_open);

        tv_flashAuto.setOnClickListener(this);
        tv_flashClose.setOnClickListener(this);
        tv_flashKeepOpen.setOnClickListener(this);
        tv_flashOpen.setOnClickListener(this);

        mSurfaceHolder = mSurfaceView.getHolder();
        mSurfaceHolder.setKeepScreenOn(true);
        // mSurfaceView????????????
        mSurfaceHolder.addCallback(new SurfaceHolder.Callback() {
            @Override
            public void surfaceCreated(SurfaceHolder holder) { //SurfaceView??????
                // ?????????Camera
                //?????????????????????
                if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.M) {
                    if(checkSelfPermission(Manifest.permission.CAMERA)!=PackageManager.PERMISSION_GRANTED
                            ||checkSelfPermission(Manifest.permission.WRITE_EXTERNAL_STORAGE)!=PackageManager.PERMISSION_GRANTED){
                        requestPermissions(new String[]{Manifest.permission.CAMERA,Manifest.permission.WRITE_EXTERNAL_STORAGE},1);
                        findViewById(R.id.btn_control).setClickable(false);
                    }else {
                        initCamera2();
                    }
                }else{
                    initCamera2();
                }
            }

            @Override
            public void surfaceChanged(SurfaceHolder holder, int format, int width, int height) {
            }

            @Override
            public void surfaceDestroyed(SurfaceHolder holder) { //SurfaceView??????
                // ??????Camera??????
                if (null != mCameraDevice) {
                    mCameraDevice.close();
                    mCameraDevice = null;
                }
            }
        });

        findViewById(R.id.btn_control).setOnClickListener(v -> takePicture());

        //?????????????????????
        iv_flash.setOnClickListener(v -> {
            Animation anim = AnimationUtils.loadAnimation(this, R.anim.flash_in);
            mFlashLayout.setAnimation(anim);
            mFlashLayout.setVisibility(View.VISIBLE);
        });

        //????????????
        iv_thumb.setOnClickListener(v -> {
            File file=new File(Environment.getExternalStoragePublicDirectory(Environment.DIRECTORY_DCIM)+"/temp.jpg");
            Intent it =new Intent(Intent.ACTION_VIEW);
            Uri mUri = Uri.parse("file://"+file.getPath());
            it.setDataAndType(mUri, "image/*");
            startActivity(it);
        });

        //???????????????
        iv_changeCamera.setOnClickListener(v->changeCamera());

    }

    @SuppressLint({"MissingPermission", "NewApi"})
    private void changeCamera(){
        try {
            //???????????????????????????
            if(mCameraDevice!=null) {
                mCameraDevice.close();
                mCameraDevice = null;
            }
            mCameraID^=1;
            mCameraManager.openCamera(mCameraID+"", stateCallback, mainHandler);
        } catch (CameraAccessException e) {
            e.printStackTrace();
        }
    }

    //????????????,??????openCV?????????????????????area interpolation???
    private Bitmap scaleImage(Bitmap bitmap, int width, int height)
    {

        Mat src = new Mat();
        Mat dst = new Mat();
        Utils.bitmapToMat(bitmap, src);
        //new Size(width, height)
        Imgproc.resize(src, dst, new Size(width,height),0,0,Imgproc.INTER_AREA);
        Bitmap bitmap1 = Bitmap.createBitmap(dst.cols(),dst.rows(),Bitmap.Config.RGB_565);
        Utils.matToBitmap(dst, bitmap1);
        return bitmap1;
    }

    @SuppressLint("NewApi")
    private void initCamera2() {
        findViewById(R.id.btn_control).setClickable(true);

        HandlerThread handlerThread = new HandlerThread("Camera2");
        handlerThread.start();
        childHandler = new Handler(handlerThread.getLooper());
        mainHandler = new Handler(getMainLooper());


        mCameraID = BACK_CAMERA;//????????????
        Log.d(TAG, "initCamera2: "+ CameraCharacteristics.LENS_FACING_BACK);
        mImageReader = ImageReader.newInstance(1080, 1920, ImageFormat.JPEG,1);
        mImageReader.setOnImageAvailableListener(new ImageReader.OnImageAvailableListener() {
            //???????????????????????????????????????????????? ?????????????????????
            @Override
            public void onImageAvailable(ImageReader reader) {
//                mCameraDevice.close();
//                mSurfaceView.setVisibility(View.INVISIBLE);
                // ????????????????????????
                Image image = reader.acquireNextImage();
                ByteBuffer buffer = image.getPlanes()[0].getBuffer();
                byte[] bytes = new byte[buffer.remaining()];
                buffer.get(bytes);//??????????????????????????????
                //??????

                Bitmap bitmap = BitmapFactory.decodeByteArray(bytes, 0, bytes.length);

                if (bitmap != null) {
                    if(mCameraID == FRONT_CAMERA){
                        //?????????????????????????????????180???
                        bitmap = adjustPhotoRotation(bitmap,180);
                    }
                    iv_thumb.setImageBitmap(bitmap);
                    writeToFile(bitmap);
                }

                image.close();

                //????????????????????????????????? 128*128
                Bitmap displayBitmap = scaleImage(bitmap,128,128);
                //????????????????????????????????? 64*64
                Bitmap bitmapForPredit = scaleImage(bitmap,64,64);

//                //??????????????????????????????
                //MediaStore.Images.Media.insertImage(getContentResolver(), bitmapForPredit, "title", "description");

                //????????????
                classifier = new Classifier(getAssets(),MODEL_FILE);
                ArrayList<String> result = classifier.predict(bitmapForPredit);
                //????????????
                Bundle bundle = new Bundle();
                bundle.putParcelable("image",displayBitmap);
                bundle.putStringArrayList("recognize_result",result);
                Intent intent = new Intent(CameraActivity.this, DisplayResult.class);
                intent.putExtras(bundle);
                startActivity(intent);
            }

        }, mainHandler);
        //?????????????????????
        mCameraManager = (CameraManager) getSystemService(Context.CAMERA_SERVICE);

        try {
            if (ActivityCompat.checkSelfPermission(this, Manifest.permission.CAMERA) != PackageManager.PERMISSION_GRANTED) {
                return;
            }

            //???????????????
            mCameraManager.openCamera(mCameraID+"", stateCallback, mainHandler);
        } catch (CameraAccessException e) {
            e.printStackTrace();
        }
    }

    Bitmap adjustPhotoRotation(Bitmap bm, final int orientationDegree)
    {
        Matrix m = new Matrix();
        m.setRotate(orientationDegree, (float) bm.getWidth() / 2, (float) bm.getHeight() / 2);

        try {
            Bitmap bm1 = Bitmap.createBitmap(bm, 0, 0, bm.getWidth(), bm.getHeight(), m, true);
            return bm1;
        } catch (OutOfMemoryError ex) {
        }
        return null;
    }

    private void writeToFile(Bitmap bitmap) {
        File file = new File(Environment.getExternalStoragePublicDirectory(Environment.DIRECTORY_DCIM).getAbsolutePath()+"/temp.jpg");
        try {
            OutputStream os = new FileOutputStream(file);

            bitmap.compress(Bitmap.CompressFormat.JPEG,100,os);

        } catch (FileNotFoundException e) {
            e.printStackTrace();
        }
    }

    /**
     * ????????????
     */
    @RequiresApi(api = Build.VERSION_CODES.LOLLIPOP)
    private void takePreview() {
        try {
            // ?????????????????????CaptureRequest.Builder
            mPreviewRequestBuilder = mCameraDevice.createCaptureRequest(CameraDevice.TEMPLATE_PREVIEW);
            // ???SurfaceView???surface??????CaptureRequest.Builder?????????
            mPreviewRequestBuilder.addTarget(mSurfaceHolder.getSurface());
            // ??????CameraCaptureSession?????????????????????????????????????????????????????????
            mCameraDevice.createCaptureSession(Arrays.asList(mSurfaceHolder.getSurface(), mImageReader.getSurface()), new CameraCaptureSession.StateCallback(){
                @Override
                public void onConfigured(CameraCaptureSession cameraCaptureSession) {
                    if (null == mCameraDevice) return;
                    // ???????????????????????????????????????????????????
                    mCameraCaptureSession = cameraCaptureSession;
                    try {
                        // ????????????
                        mPreviewRequestBuilder.set(CaptureRequest.CONTROL_AF_MODE, CaptureRequest.CONTROL_AF_MODE_CONTINUOUS_PICTURE);
                        // ???????????????
                        mPreviewRequestBuilder.set(CaptureRequest.CONTROL_AE_MODE, CaptureRequest.FLASH_MODE_OFF);
                        // ????????????
                        CaptureRequest previewRequest = mPreviewRequestBuilder.build();
                        mCameraCaptureSession.setRepeatingRequest(previewRequest, null, childHandler);
                    } catch (CameraAccessException e) {
                        e.printStackTrace();
                    }
                }

                @Override
                public void onConfigureFailed(CameraCaptureSession cameraCaptureSession) {
                    Toast.makeText(CameraActivity.this, "????????????", Toast.LENGTH_SHORT).show();
                }
            }, childHandler);
        } catch (CameraAccessException e) {
            e.printStackTrace();
        }
    }

    @SuppressLint("NewApi")
    public void changeFlash(int type){
        try {
            CaptureRequest.Builder requestBuilder = mCameraDevice.createCaptureRequest(CameraDevice.TEMPLATE_PREVIEW);
            requestBuilder.addTarget(mSurfaceHolder.getSurface());
            if(type == FLASH_ON)
                requestBuilder.set(CaptureRequest.CONTROL_AE_MODE, CaptureRequest.CONTROL_AE_MODE_ON_AUTO_FLASH);
            else if(type == FLASH_OFF)
                requestBuilder.set(CaptureRequest.CONTROL_AE_MODE, CaptureRequest.FLASH_MODE_OFF);
            mCameraCaptureSession.setRepeatingRequest(requestBuilder.build(),null,childHandler);
        } catch (CameraAccessException e) {
            e.printStackTrace();
        }
    }

    /**
     * ?????????????????????
     */
    private CameraDevice.StateCallback stateCallback = new CameraDevice.StateCallback() {
        @TargetApi(Build.VERSION_CODES.LOLLIPOP)
        @Override
        public void onOpened(CameraDevice camera) {//???????????????
            mCameraDevice = camera;
            //????????????
            takePreview();
        }

        @Override
        public void onDisconnected(CameraDevice camera) {//???????????????
            if (null != mCameraDevice) {
                mCameraDevice.close();
                CameraActivity.this.mCameraDevice = null;
            }
        }

        @Override
        public void onError(CameraDevice camera, int error) {//????????????
            Toast.makeText(CameraActivity.this, "?????????????????????", Toast.LENGTH_SHORT).show();
        }
    };

    /**
     * ??????
     */
    @SuppressLint("NewApi")
    private void takePicture() {
        if (mCameraDevice == null) return;
        // ?????????????????????CaptureRequest.Builder
        final CaptureRequest.Builder captureRequestBuilder;
        try {
            captureRequestBuilder = mCameraDevice.createCaptureRequest(CameraDevice.TEMPLATE_STILL_CAPTURE);
            // ???imageReader???surface??????CaptureRequest.Builder?????????
            captureRequestBuilder.addTarget(mImageReader.getSurface());
            // ????????????
            captureRequestBuilder.set(CaptureRequest.CONTROL_AF_MODE, CaptureRequest.CONTROL_AF_MODE_CONTINUOUS_PICTURE);
            // ????????????
            captureRequestBuilder.set(CaptureRequest.CONTROL_AE_MODE, CaptureRequest.CONTROL_AE_MODE_ON_AUTO_FLASH);
            // ??????????????????
            int rotation = getWindowManager().getDefaultDisplay().getRotation();
            // ?????????????????????????????????????????????
            captureRequestBuilder.set(CaptureRequest.JPEG_ORIENTATION, ORIENTATIONS.get(rotation));
            //??????
            CaptureRequest mCaptureRequest = captureRequestBuilder.build();
            mCameraCaptureSession.capture(mCaptureRequest, null, childHandler);
        } catch (CameraAccessException e) {
            e.printStackTrace();
        }
    }

    @Override
    public void onClick(View v) {
        FlashStrategy strategy = null;
        switch (v.getId()){
            case R.id.tv_flash_close:
                strategy = new CloseStrategy();
                tv_flashStatus.setText("??????");
                break;
            case R.id.tv_flash_auto:
                tv_flashStatus.setText("??????");
                strategy = new AutoStrategy();
                break;
            case R.id.tv_flash_open:
                tv_flashStatus.setText("??????");
                strategy = new OpenStrategy();
                break;
            case R.id.tv_flash_keep_open:
                tv_flashStatus.setText("??????");
                strategy = new KeepOpenStrategy();
                break;
        }
        //????????????????????? ???????????????
        if(strategy!=null) {
            //????????????????????????
            clearAllFlashTextBackground(v.getId());
            //????????????
            closeFlashLayout();
            //?????????????????????
            strategy.setCaptureRequest(mPreviewRequestBuilder, mCameraCaptureSession, childHandler);
        }

    }

    public void closeFlashLayout(){
        Animation anim = AnimationUtils.loadAnimation(this, R.anim.flash_out);
        mFlashLayout.setAnimation(anim);
        mFlashLayout.setVisibility(View.GONE);
    }

    public void clearAllFlashTextBackground(int id){
        tv_flashClose.setBackground(null);
        tv_flashOpen.setBackground(null);
        tv_flashKeepOpen.setBackground(null);
        tv_flashAuto.setBackground(null);
        findViewById(id).setBackground(getResources().getDrawable(R.drawable.flash_text_shape));
    }



}
