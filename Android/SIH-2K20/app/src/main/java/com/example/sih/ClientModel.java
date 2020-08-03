package com.example.sih;

import java.io.Serializable;
import java.util.Comparator;

public class ClientModel implements Serializable
{
    private String url;
    int status;
    String url2;
    Float fov,gamma;

    public ClientModel(String url, int status, String url2, Float fov, Float gamma) {
        this.url = url;
        this.status = status;
        this.url2 = url2;
        this.fov = fov;
        this.gamma = gamma;
    }

    public String getUrl() {
        return url;
    }

    public void setUrl(String url) {
        this.url = url;
    }

    public int getStatus() {
        return status;
    }

    public void setStatus(int status) {
        this.status = status;
    }

    public String getUrl2() {
        return url2;
    }

    public void setUrl2(String url2) {
        this.url2 = url2;
    }

    public Float getFov() {
        return fov;
    }

    public void setFov(Float fov) {
        this.fov = fov;
    }

    public Float getGamma() {
        return gamma;
    }

    public void setGamma(Float gamma) {
        this.gamma = gamma;
    }
}