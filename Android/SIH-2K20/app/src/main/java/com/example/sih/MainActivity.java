package com.example.sih;

import androidx.annotation.NonNull;
import androidx.appcompat.app.AppCompatActivity;
import androidx.core.app.ActivityCompat;
import androidx.core.content.ContextCompat;

import android.Manifest;
import android.app.ProgressDialog;
import android.content.ContentResolver;
import android.content.Context;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.hardware.Camera;
import android.net.Uri;
import android.os.Bundle;
import android.provider.MediaStore;
import android.util.Log;
import android.view.View;
import android.webkit.MimeTypeMap;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.Toast;

import com.google.android.gms.tasks.OnFailureListener;
import com.google.android.gms.tasks.OnSuccessListener;
import com.google.firebase.database.DatabaseReference;
import com.google.firebase.database.FirebaseDatabase;
import com.google.firebase.storage.FirebaseStorage;
import com.google.firebase.storage.OnProgressListener;
import com.google.firebase.storage.StorageReference;
import com.google.firebase.storage.UploadTask;

import java.io.ByteArrayOutputStream;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.security.Policy;
import java.util.Random;

import retrofit2.Call;
import retrofit2.Callback;
import retrofit2.Response;

public class MainActivity extends AppCompatActivity {
    Button camerab, galleryb, uploadb;
    ImageView imageb;
    //uri to store file
     Uri filePath;
     String url;
    String responseurl;
  //  Camera mcamera;
    //int focul_length;
    //Policy.Parameters params;
  //  File mFile;

///    public int PICTURE_ACTIVITY_CODE = 1;
  //  public String FILENAME = "sdcard/photo.jpg";
   // Camera.Parameters cameraParameters;

    ClientAPI clientAPI = Utils.getClientAPI();

    //firebase objects
    private StorageReference storageReference;
    private DatabaseReference mDatabase;
//    private Firebase mRef;
//    private StorageReference storageRef;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
//        mcamera = Camera.open();
//        cameraParameters = mcamera.getParameters();
//        Camera.CameraInfo myinfo = new Camera.CameraInfo();
//        float l=cameraParameters.getFocalLength(); // Here its creating Null Pointer Exception
//        Toast.makeText(MainActivity.this, String.valueOf(l), Toast.LENGTH_SHORT).show();
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

        camerab.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                final int CHOOSE_CAMERA = 232;
                Intent cameraIntent = new Intent(MediaStore.ACTION_IMAGE_CAPTURE);

                startActivityForResult(cameraIntent, 0);

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
                uploadFile();

            }
        });

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

                    Log.e("selectedImageUri "," = " + selectedImageUri);
                    if(selectedImageUri!=null){

                        Bitmap bmp= BitmapFactory.decodeStream(getContentResolver().openInputStream(selectedImageUri));
                        Log.e("bmp "," = " + bmp);
                        imageb.setImageBitmap(bmp);
                        Log.e("bmp ", " Displaying Imageview WIth Bitmap !!!!  = ");
                    } else {
                        // If selectedImageUri is null check extras for bitmap

                        Bitmap bmp = (Bitmap) imageReturnedIntent.getExtras().get("data");
                       // Uri thumbUri = getImageUri(this, bmp);
                      //  filePath=getImageUri(this,bmp);

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


    private void uploadFile() {
        //checking if file is available
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
                            progressDialog.dismiss();

                            //displaying success toast
                            Toast.makeText(getApplicationContext(), "File Uploaded ", Toast.LENGTH_LONG).show();


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
                                    Log.e("urllll",url);

                                    clientAPI.search(url, (int) 70.00,200).enqueue(new Callback<ResponseClient>() {
                                        @Override
                                        public void onResponse(Call<ResponseClient> call, Response<ResponseClient> response) {
                                            Log.e("vjh","hhb");
                                            if (response.isSuccessful()) {
                                                 responseurl=response.body().getUrl();
                                                Intent i = new Intent(MainActivity.this, Calculation.class);
                                                i.putExtra("urlsend",responseurl);
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
