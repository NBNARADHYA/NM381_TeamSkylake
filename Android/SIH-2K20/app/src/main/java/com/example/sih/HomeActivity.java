package com.example.sih;

import androidx.annotation.Nullable;
import androidx.appcompat.app.AppCompatActivity;

import android.content.Intent;
import android.content.pm.PackageInfo;
import android.content.pm.PackageManager;
import android.content.pm.Signature;
import android.os.Bundle;
import android.util.Base64;
import android.util.Log;
import android.widget.Toast;

import com.facebook.accountkit.AccessToken;
import com.facebook.accountkit.AccountKit;
import com.facebook.accountkit.AccountKitLoginResult;
import com.facebook.accountkit.ui.AccountKitActivity;
import com.facebook.accountkit.ui.AccountKitConfiguration;
import com.facebook.accountkit.ui.LoginType;
import com.google.android.gms.common.internal.service.Common;

import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;

public class HomeActivity extends AppCompatActivity {
   // private static final int APP_REQUEST_CODE = 17;

    void loginUser(){
        final Intent intent=new Intent(HomeActivity.this,AccountKitActivity.class);
        AccountKitConfiguration.AccountKitConfigurationBuilder configurationBuilder=new AccountKitConfiguration.AccountKitConfigurationBuilder(LoginType.PHONE,
                AccountKitActivity.ResponseType.TOKEN);
        intent.putExtra(AccountKitActivity.ACCOUNT_KIT_ACTIVITY_CONFIGURATION,configurationBuilder.build());
        startActivityForResult(intent,17);

    }
    @Override
    protected void onActivityResult(int requestCode, int resultCode, @Nullable Intent data) {
        super.onActivityResult(requestCode, resultCode, data);
        if(requestCode==17)
        {
            AccountKitLoginResult loginResult= data.getParcelableExtra(AccountKitLoginResult.RESULT_KEY);
            if(loginResult.getError()!=null)
            {
                Toast.makeText(HomeActivity.this, ""+loginResult.getError().getErrorType().getMessage(), Toast.LENGTH_SHORT).show();
            }
            else if(loginResult.wasCancelled())
            {
                Toast.makeText(this, "Login Cancelled", Toast.LENGTH_SHORT).show();
            }
            else {
                Intent intent=new Intent(HomeActivity.this,MainActivity.class);
               // intent.putExtra(Common.IS_LOGIN,true);
                startActivity(intent);
                finish();

            }

        }
    }

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);


        AccessToken accessToken= AccountKit.getCurrentAccessToken();
        if(accessToken!=null)//already logged in
        {  Intent intent=new Intent(HomeActivity.this,MainActivity.class);
           // intent.putExtra(Common.IS_LOGIN,true);
            startActivity(intent);
            finish();
        }
        else
        {   loginUser();
           // setContentView(R.layout.activity_main);

        }

    }

    private void printKetHash() {
        try {
            PackageInfo packageInfo = getPackageManager().getPackageInfo(getPackageName(), PackageManager.GET_SIGNATURES);

            for(Signature signature : packageInfo.signatures)
            {
                MessageDigest md=MessageDigest.getInstance("SHA");
                md.update(signature.toByteArray());
                Log.e("HAshKEy", Base64.encodeToString(md.digest(),Base64.DEFAULT));
            }
        } catch (PackageManager.NameNotFoundException e) {
            e.printStackTrace();
        } catch (NoSuchAlgorithmException e) {
            e.printStackTrace();
        }
        ;

    }
}
