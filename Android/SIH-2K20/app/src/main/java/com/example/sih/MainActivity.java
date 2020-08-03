package com.example.sih;

import androidx.annotation.NonNull;
import androidx.appcompat.app.AppCompatActivity;
import androidx.core.app.ActivityCompat;
import androidx.core.content.ContextCompat;
import androidx.loader.content.CursorLoader;


import android.Manifest;
import android.app.Activity;
import android.app.ProgressDialog;
import android.content.ContentResolver;
import android.content.Context;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.database.Cursor;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.hardware.Camera;
import android.hardware.Sensor;
import android.hardware.SensorEvent;
import android.hardware.SensorEventListener;
import android.hardware.SensorManager;
import android.net.ConnectivityManager;
import android.net.NetworkInfo;
import android.net.Uri;
import android.os.AsyncTask;
import android.os.Bundle;
import android.os.Environment;
import android.provider.MediaStore;
import android.util.Base64;
import android.util.Log;
import android.view.View;
import android.webkit.MimeTypeMap;
import android.widget.Button;
import android.widget.EditText;
import android.widget.ImageView;
import android.widget.TextView;
import android.widget.Toast;

import com.google.android.gms.tasks.OnFailureListener;
import com.google.android.gms.tasks.OnSuccessListener;
import com.google.firebase.database.DatabaseReference;
import com.google.firebase.database.FirebaseDatabase;
import com.google.firebase.storage.FirebaseStorage;
import com.google.firebase.storage.OnProgressListener;
import com.google.firebase.storage.StorageReference;
import com.google.firebase.storage.UploadTask;

import org.apache.http.HttpEntity;
import org.apache.http.HttpResponse;
import org.apache.http.client.HttpClient;
import org.apache.http.client.methods.HttpPost;
import org.apache.http.entity.StringEntity;
import org.apache.http.impl.client.DefaultHttpClient;
import org.apache.http.util.EntityUtils;

import java.io.BufferedReader;
import java.io.ByteArrayOutputStream;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.security.Policy;
import java.text.DateFormat;
import java.text.SimpleDateFormat;
import java.util.Date;
import java.util.Random;

import retrofit2.Call;
import retrofit2.Callback;
import retrofit2.Response;

public class MainActivity extends AppCompatActivity {
    Button camerab, galleryb, uploadb;
    ImageView imageb;
    //uri to store file
     Uri filePath;
     String url,url1;
    String responseurl;
    float x2;
    float x,y,z;
    TextView xyza,anglea;
    private String mCameraFileName;


    //  Camera mcamera;

    TextView tvIsConnected;
    EditText etName,etCountry,etTwitter;
    Button btnPost;
    static TextView tvp;
    String disp= "Zooooooooooo";
    ClientAPI clientAPI = Utils.getClientAPI();


    private StorageReference storageReference;
    private DatabaseReference mDatabase;


    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        xyza=findViewById(R.id.xyz);
        anglea=findViewById(R.id.angl);
        x2=(float)1.1568369;


//        mcamera = Camera.open();
//        cameraParameters = mcamera.getParameters();
//        Camera.CameraInfo myinfo = new Camera.CameraInfo();
//        float l=cameraParameters.getFocalLength(); // Here its creating Null Pointer Exception
//        Toast.makeText(MainActivity.this, String.valueOf(l), Toast.LENGTH_SHORT).show();

       // findhorizontal();
        //Toast.makeText(MainActivity.this, x+" "+y+" "+z, Toast.LENGTH_SHORT).show();

        //findxyz();

        camerab = findViewById(R.id.camera);
        galleryb = findViewById(R.id.gallery);
        imageb = findViewById(R.id.image);
        uploadb = findViewById(R.id.upload);
        storageReference = FirebaseStorage.getInstance().getReference();
        mDatabase = FirebaseDatabase.getInstance().getReference(Constants.DATABASE_PATH_UPLOADS);
        if (ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA)
                == PackageManager.PERMISSION_DENIED) {
            ActivityCompat.requestPermissions(MainActivity.this, new String[]{Manifest.permission.CAMERA}, 3);

        }
        SensorManager sensorManager = (SensorManager) this.getSystemService(SENSOR_SERVICE);

        final float[] mValuesMagnet      = new float[3];
        final float[] mValuesAccel       = new float[3];
        final float[] mValuesOrientation = new float[3];
        final float[] mRotationMatrix    = new float[9];

//        final Button btn_valider = (Button) findViewById(R.id.button2);
//        final TextView txt1 = (TextView) findViewById(R.id.textView3);
        final SensorEventListener mEventListener = new SensorEventListener() {
            public void onAccuracyChanged(Sensor sensor, int accuracy) {
            }

            public void onSensorChanged(SensorEvent event) {
                // Handle the events for which we registered
                switch (event.sensor.getType()) {
                    case Sensor.TYPE_ACCELEROMETER:
                        System.arraycopy(event.values, 0, mValuesAccel, 0, 3);
                        break;

                    case Sensor.TYPE_MAGNETIC_FIELD:
                        System.arraycopy(event.values, 0, mValuesMagnet, 0, 3);
                        break;
                }
            };
        };
        // You have set the event lisetner up, now just need to register this with the
        // sensor manager along with the sensor wanted.
        setListners(sensorManager, mEventListener);

        camerab.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                final int CHOOSE_CAMERA = 232;
//                Intent cameraIntent = new Intent(MediaStore.ACTION_IMAGE_CAPTURE);
//                startActivityForResult(cameraIntent, 0);


                Intent intent = new Intent();
                // Picture from camera
                intent.setAction(MediaStore.ACTION_IMAGE_CAPTURE);

                // This is not the right way to do this, but for some reason, having
                // it store it in
                // MediaStore.Images.Media.EXTERNAL_CONTENT_URI isn't working right.

//                Date date = new Date();
//                DateFormat df = new SimpleDateFormat("-mm-ss");
//
//                String newPicFile = "Bild"+ df.format(date) + ".jpg";
//                String outPath = "/sdcard/" + newPicFile;
//                File outFile = new File(outPath);
//
//                mCameraFileName = outFile.toString();
//                Uri outuri = Uri.fromFile(outFile);
//                intent.putExtra(MediaStore.EXTRA_OUTPUT, outuri);


                startActivityForResult(intent, 0);



                SensorManager.getRotationMatrix(mRotationMatrix, null, mValuesAccel, mValuesMagnet);
                SensorManager.getOrientation(mRotationMatrix, mValuesOrientation);
                final CharSequence test;
                x= mValuesOrientation[0] ;
                y=mValuesOrientation[1];
                z=mValuesOrientation[2];



            }
        });

        galleryb.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                Intent pickPhoto = new Intent(Intent.ACTION_PICK,
                        android.provider.MediaStore.Images.Media.EXTERNAL_CONTENT_URI);
                startActivityForResult(pickPhoto, 1);//one can be replaced with any action code

            }
        });
        uploadb.setOnClickListener(new View.OnClickListener() {

            @Override
            public void onClick(View view) {
                //Toast.makeText(MainActivity.this, x+" "+y+" "+z+"   "+x2, Toast.LENGTH_LONG).show();

                uploadFile();



            }
        });

    }


    private Uri mImageUri = null;

    private static final  int GALLERY_REQUEST =1;

    private static final int CAMERA_REQUEST_CODE=1;

    private StorageReference mStorage;

    public void setListners(SensorManager sensorManager, SensorEventListener mEventListener)
    {
        sensorManager.registerListener(mEventListener, sensorManager.getDefaultSensor(Sensor.TYPE_ACCELEROMETER),
                SensorManager.SENSOR_DELAY_NORMAL);
        sensorManager.registerListener(mEventListener, sensorManager.getDefaultSensor(Sensor.TYPE_MAGNETIC_FIELD),
                SensorManager.SENSOR_DELAY_NORMAL);
    }






    public void anglee(View v){
      //  Intent i=new Intent(MainActivity.this,fov.class);
       // startActivity(i);
//        this.x2=getIntent().getFloatExtra("horizon", (float) 1.15682);
//        Log.e("horizontal", String.valueOf(x2));
        anglea.setText("           Horizontal : 1.1568369");
        xyza.setText("            x : -0.55600554 y:-0.48234528, z:0.031622503");
    }

    //    @Override
//    protected void onStart() {
//        super.onStart();
//        googleApiClient.connect();
//        //Firebase
//        mRef = new Firebase("link to firebase account");
//        FirebaseStorage storage = FirebaseStorage.getInstance();
//        storageRef = storage.getReferenceFromUrl("link to storage");
//
//    }


    protected void onActivityResult(int requestCode, int resultCode, Intent imageReturnedIntent) {
        super.onActivityResult(requestCode, resultCode, imageReturnedIntent);

        switch (requestCode) {
            case 0:
                try{
                    Uri selectedImageUri  = imageReturnedIntent.getData();


                    filePath=selectedImageUri;
                    Log.e("selectedImageUri "," = " + selectedImageUri);
                    if(selectedImageUri!=null){

                        Bitmap bmp= BitmapFactory.decodeStream(getContentResolver().openInputStream(selectedImageUri));
                        Log.e("bmp "," = " + bmp);
                        imageb.setImageBitmap(bmp);
                        String result = MediaStore.Images.Media.insertImage(getContentResolver(), bmp, "", "");

                        filePath = Uri.parse(result);
                        Log.e("bmp ", " Displaying Imageview WIth Bitmap !!!!  = ");
                    } else {
                        // If selectedImageUri is null check extras for bitmap

                        Bitmap bmp = (Bitmap) imageReturnedIntent.getExtras().get("data");
                       // Uri thumbUri = getImageUri(this, bmp);
                      // filePath=getImageUri(this,bmp);
                        String result = MediaStore.Images.Media.insertImage(getContentResolver(), bmp, "", "");

                        filePath = Uri.parse(result);
                        imageb.setImageBitmap(bmp);
                    }

                }
                catch (FileNotFoundException fe)
                {
                    fe.printStackTrace();
                }

                break;
            case 1:
                if (resultCode == RESULT_OK) {
                    filePath = imageReturnedIntent.getData();
                    try {
                        Bitmap bitmap = MediaStore.Images.Media.getBitmap(getContentResolver(), filePath);
                        imageb.setImageBitmap(bitmap);
                    } catch (IOException e) {
                        e.printStackTrace();
                    }
                }
                break;
        }
    }
//    public Uri getImageUri(Context inContext, Bitmap inImage) {
//        ByteArrayOutputStream bytes = new ByteArrayOutputStream();
//        inImage.compress(Bitmap.CompressFormat.JPEG, 100, bytes);
//        String path = MediaStore.Images.Media.insertImage(inContext.getContentResolver(), inImage, "Title", null);
//        return Uri.parse(path);
//    }
    public String getFileExtension(Uri uri) {
        ContentResolver cR = getContentResolver();
        MimeTypeMap mime = MimeTypeMap.getSingleton();
        return mime.getExtensionFromMimeType(cR.getType(uri));
    }
    private class HttpAsyncTask extends AsyncTask<String, Void, String> {
        @Override
        protected String doInBackground(String... urls) {


            return POST(urls[0]);
        }
        // onPostExecute displays the results of the AsyncTask.
        @Override
        protected void onPostExecute(String result) {
            Toast.makeText(getBaseContext(), "Data Sent!", Toast.LENGTH_LONG).show();
        }
    }
//    private String getRealPathFromURI(Uri contentUri) {
//        String[] proj = { MediaStore.Images.Media.DATA };
//        CursorLoader loader = new CursorLoader(this, contentUri, proj, null, null, null);
//        Cursor cursor = loader.loadInBackground();
//        int column_index = cursor.getColumnIndexOrThrow(MediaStore.Images.Media.DATA);
//        cursor.moveToFirst();
//        String result = cursor.getString(column_index);
//        cursor.close();
//        return result;
//    }


    public String POST(String url)
    {
        Log.i("MINION", "inside POST()");

        InputStream inputStream = null;
        String result = "";
        byte[] buffer;
        int bufferSize = 1 * 1024 * 1024;
        Bitmap bm;
//        File file2 = new File(filePath.getPath());//create path from uri
//        final String[] split = file2.getPath().split(":");//split the path.
//        String imagePath = split[1];//assign it to a string(your choice).
        String imagePath=FileUtils.getPath(this,filePath);
        String encodedImage = null;
        File file = new File(imagePath);
        String temp = "Sanket Agarwal";
        int responseCode=0;
        String responseMessage = "";
        try {

            Log.i("MINION", "inside try block");

            // 1. create HttpClient
            HttpClient httpclient = new DefaultHttpClient();

            // 2. make POST request to the given URL
            HttpPost httpPost = new HttpPost(url);


            //Check if file exists
            if (!file.isFile()) {
                Log.e("uploadFile", "Source File not exist :"+imagePath);
            }

            else
            {
                Log.i("MINION", "image file found");
                FileInputStream fileInputStream = new FileInputStream(file);
                buffer = new byte[bufferSize];
                bm = BitmapFactory.decodeStream(fileInputStream);
                ByteArrayOutputStream baos = new ByteArrayOutputStream();
                buffer = baos.toByteArray();	//Image->Byte Array


                //Byte Array to base64 image string
                Toast.makeText(this, "working", Toast.LENGTH_SHORT).show();
                encodedImage = Base64.encodeToString(buffer, Base64.DEFAULT);
                Log.i("MINION", "image converted to Base 64 string");

                //Converting encodedImage to String Entity
                StringEntity se = new StringEntity(encodedImage);
                //StringEntity se = new StringEntity(temp);
                Log.i("string",se.toString());
                Log.i("MINION", "encodedImage to StringEntity");

                //httpPost Entity
                httpPost.setEntity(se);
                // 7. Set some headers to inform server about the type of the content
                httpPost.setHeader("Accept", "multipart/form-data");
                httpPost.setHeader("Content-type", "multipart/form-data");

                // 8. Execute POST request to the given URL
                HttpResponse httpResponse = httpclient.execute(httpPost);
                Log.i("MINION", "Post request successful");

                // 9. receive response as inputStream

                inputStream = httpResponse.getEntity().getContent();

                HttpEntity entity = httpResponse.getEntity();

                String responseText = EntityUtils.toString(entity);

                responseCode = httpResponse.getStatusLine().getStatusCode();
                System.out.println("Response Code: " + responseCode);

                responseMessage = EntityUtils.toString(httpResponse.getEntity());
                System.out.println("Response Message: " + responseText);


	           /* // 10. convert inputstream to string
	            if(inputStream != null)
	            {
	            	Log.i("MINION", "Converting Response to String");
	            	result = convertInputStreamToString(inputStream);
	            	Log.i("MINION", "Response to String Sucess");
	            }
	            else
	            	result = "Did not work!";

	            */
            }


        } catch (Exception e) {
            Log.d("InputStream", e.toString());
        }

        // 11. return result
        result = "Response Code: " + responseCode + "Response Message: " + responseMessage + result;
        System.out.println("Result: " + result);
        Log.i("result",result);
        return result;
    }

    public boolean isConnected(){
        ConnectivityManager connMgr = (ConnectivityManager) getSystemService(Activity.CONNECTIVITY_SERVICE);
        NetworkInfo networkInfo = connMgr.getActiveNetworkInfo();
        if (networkInfo != null && networkInfo.isConnected())
            return true;
        else
            return false;
    }

    private static String convertInputStreamToString(InputStream inputStream) throws IOException{
        Log.i("MINION", "Inside convertINputStreamToString");
        BufferedReader bufferedReader = new BufferedReader( new InputStreamReader(inputStream));
        String line = "";
        String result = "";
        while((line = bufferedReader.readLine()) != null)
            result += line;

        inputStream.close();
        return result;

    }

    private void uploadFile() {
//        new HttpAsyncTask().execute("http://34.69.240.165:5000/upload");







    //    checking if file is available
        if (filePath != null) {
            //displaying progress dialog while image is uploading
            final ProgressDialog progressDialog = new ProgressDialog(this);
            progressDialog.setTitle("Uploading");
            progressDialog.show();

            //getting the storage reference
            final StorageReference sRef = storageReference.child(Constants.STORAGE_PATH_UPLOADS + System.currentTimeMillis() + "." + getFileExtension(filePath));

            //adding the file to reference
            sRef.putFile(filePath)
                    .addOnSuccessListener(new OnSuccessListener<UploadTask.TaskSnapshot>() {
                        @Override
                        public void onSuccess(UploadTask.TaskSnapshot taskSnapshot) {
                            //dismissing the progress dialog

                            //creating the upload object to store uploaded image details
                            Upload upload = new Upload(taskSnapshot.getMetadata().getReference().getDownloadUrl().toString());
                            String uploadId = mDatabase.push().getKey();
                            mDatabase.child(uploadId).setValue(upload);
                            sRef.getDownloadUrl().addOnSuccessListener(new OnSuccessListener<Uri>() {
                                @Override
                                public void onSuccess(Uri uri) {
                                    Uri downloadUrl = uri;
                                    //Do what you want with the url
                                   // Log.e("urlf",downloadUrl.toString());
                                     url = downloadUrl.toString();
                                    url1=url;
                                    Log.e("urllll",url);

                                    clientAPI.search(url, (int) 70.00,url1,x2,y).enqueue(new Callback<ResponseClient>() {
                                        @Override
                                        public void onResponse(Call<ResponseClient> call, Response<ResponseClient> response) {
                                            Log.e("vjh","hhb");
                                            if (response.isSuccessful()) {
                                                 responseurl=response.body().getUrl();
                                                 url1=response.body().getUrl1();
                                                progressDialog.dismiss();

                                                //displaying success toast
                                                Toast.makeText(getApplicationContext(), "File Uploaded ", Toast.LENGTH_LONG).show();


                                                Intent i = new Intent(MainActivity.this, Calculation.class);
                                                i.putExtra("urlsend",responseurl);
                                                Log.e("bhaotho",url1);
                                                i.putExtra("url1",url1);
                                                i.putExtra("horizontal",x2);
                                                i.putExtra("x",x);
                                                i.putExtra("y",y);
                                                i.putExtra("z",z);


//        Log.e("dfddddd",responseurl);
                                                startActivity(i);

                                                Log.e("respo5nse",responseurl);
                                            } else {
                                                Log.e("ddddd",response.message());
                                            }
                                        }

                                        @Override
                                        public void onFailure(Call<ResponseClient> call, Throwable t) {
                                            Log.e("Error aa gaya", t.getMessage());
                                        }
                                    });
                                }
                            });
                            //adding an upload to firebase database
//                            String uploadId = mDatabase.push().getKey();
//                            mDatabase.child(uploadId).setValue(upload);

                        }
                    })
                    .addOnFailureListener(new OnFailureListener() {
                        @Override
                        public void onFailure(@NonNull Exception exception) {
                            progressDialog.dismiss();
                            Toast.makeText(getApplicationContext(), exception.getMessage(), Toast.LENGTH_LONG).show();
                        }
                    })
                    .addOnProgressListener(new OnProgressListener<UploadTask.TaskSnapshot>() {
                        @Override
                        public void onProgress(UploadTask.TaskSnapshot taskSnapshot) {
                            //displaying the upload progress
                            double progress = (100.0 * taskSnapshot.getBytesTransferred()) / taskSnapshot.getTotalByteCount();
                            progressDialog.setMessage("Uploaded " + ((int) progress) + "%...");
                        }
                    });
        } else {
            //display an error if no file is selected
        }


    }

}
