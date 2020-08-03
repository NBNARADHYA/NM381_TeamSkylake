package com.example.sih;

import com.google.gson.annotations.Expose;
import com.google.gson.annotations.SerializedName;

import java.util.List;

public class ResponseClient
{
    @SerializedName("segmentedImagePath")
    @Expose
    private String url;
    @SerializedName("angle_plot_img_path")
    @Expose
    private String url1;
    @SerializedName("error")
    @Expose
    private int error;

    public ResponseClient(String url, String url1, int error) {
        this.url = url;
        this.url1 = url1;
        this.error = error;
    }

    public String getUrl() {
        return url;
    }

    public void setUrl(String url) {
        this.url = url;
    }

    public String getUrl1() {
        return url1;
    }

    public void setUrl1(String url1) {
        this.url1 = url1;
    }

    public int getError() {
        return error;
    }

    public void setError(int error) {
        this.error = error;
    }
}
