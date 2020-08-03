package com.example.sih;

import java.io.Serializable;
import java.util.Comparator;

public class ClientModel implements Serializable
{
    private String url;
    int status;
    String url2;

    public ClientModel(String url, int status, String url2) {
        this.url = url;
        this.status = status;
        this.url2 = url2;
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
}