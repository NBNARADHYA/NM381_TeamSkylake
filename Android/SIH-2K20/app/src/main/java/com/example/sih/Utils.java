package com.example.sih;


public class Utils
{
    private Utils(){}

    public static final String BaseUrl="http://34.69.240.165:3000/";

    public static ClientAPI getClientAPI()
    {
        return RetrofitClient.getClient(BaseUrl).create(ClientAPI.class);
    }
}
