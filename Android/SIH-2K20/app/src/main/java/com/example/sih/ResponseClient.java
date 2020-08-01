package com.example.sih;

import com.google.gson.annotations.Expose;
import com.google.gson.annotations.SerializedName;

import java.util.List;

public class ResponseClient
{
    @SerializedName("url")
    @Expose
    private String url;
    @SerializedName("angle_of_elevation")
    @Expose
    private float angle;
    @SerializedName("error")
    @Expose
    private int error;

    public ResponseClient(String url, float angle, int error) {
        this.url = url;
        this.angle = angle;
        this.error = error;
    }

    public float getAngle() {
        return angle;
    }

    public void setAngle(float angle) {
        this.angle = angle;
    }

    public String getUrl() {
        return url;
    }

    public void setUrl(String url) {
        this.url = url;
    }

    public int getError() {
        return error;
    }

    public void setError(int error) {
        this.error = error;
    }
}
