package com.example.sih;

import retrofit2.Call;
import retrofit2.http.Field;
import retrofit2.http.FormUrlEncoded;
import retrofit2.http.POST;

public interface ClientAPI {

    @POST("/upload")
    @FormUrlEncoded
    Call<ResponseClient> search(
            @Field("url") String url,
            @Field("error") int status,
            @Field ("angle_of_elevation") float angle
    );
}


/// we are trying to get elevation angle here