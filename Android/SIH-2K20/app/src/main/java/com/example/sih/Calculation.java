package com.example.sih;

import androidx.appcompat.app.AppCompatActivity;

import android.app.ProgressDialog;
import android.content.Intent;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.os.Bundle;
import android.util.Log;
import android.widget.Button;
import android.widget.ImageView;
//import com.bumptech.glide.Glide;


import com.bumptech.glide.Glide;
import com.google.firebase.database.DataSnapshot;
import com.google.firebase.database.DatabaseError;
import com.google.firebase.database.DatabaseReference;
import com.google.firebase.database.FirebaseDatabase;
import com.google.firebase.database.ValueEventListener;

import java.io.IOException;
import java.net.MalformedURLException;
import java.net.URL;
import java.util.ArrayList;
import java.util.List;

public class Calculation extends AppCompatActivity {
    private DatabaseReference mDatabase;

  String url;
  ImageView image;
  Button angleb,datab;
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_calculation);
        String url=getIntent().getStringExtra("urlsend");
        image=findViewById(R.id.responseimage);
        angleb=findViewById(R.id.angle);

        datab=findViewById(R.id.data);

        Glide.with(this).load(url).into(image);
       Log.e("ggg",url);
        Log.e("gggg","aagya");




    }
}
