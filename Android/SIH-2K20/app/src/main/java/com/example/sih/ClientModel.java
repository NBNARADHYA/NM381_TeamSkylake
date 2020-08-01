package com.example.sih;

import java.io.Serializable;
import java.util.Comparator;

public class ClientModel implements Serializable
{
    private String url;
    int status;
    float angle;

    public ClientModel(String url, int status, float angle) {
        this.url = url;
        this.status = status;
        this.angle = angle;
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

    public int getStatus() {
        return status;
    }

    public void setStatus(int status) {
        this.status = status;
    }
}