package client

import (
	"bytes"
	"coordinator/logger"
	"encoding/json"
	"io/ioutil"
	"net/http"
	"strings"
	"time"
)

func Get(url string) string {

	logger.Do.Println("send get requests to ", url)

	resp, err := http.Get(url)

	if err != nil {
		logger.Do.Println(err)
		panic(err)
	}

	defer resp.Body.Close()

	body, err := ioutil.ReadAll(resp.Body)

	if err != nil {
		logger.Do.Println(err)
		panic(err)
	}

	if resp.StatusCode > 200 && resp.StatusCode <= 299{
		logger.Do.Println("Get request Error ", resp.StatusCode, string(body))
		panic(err)
	}

	return string(body)
}

func PostForm(addr string, data map[string][]string) error {
	/*

		eg:
			addr: "http://localhost:9089/submit"
			data：= addr.Values {
					"name": {"John Doe"},
					"occupation": {"gardener"}}

	*/
	url := "http://" + strings.TrimSpace(addr)

	logger.Do.Printf("Sending post request to url: %q", url)

	NTimes := 20

	var err error
	var resp *http.Response

	for {
		if NTimes<0{
			return err
		}

		resp, err = http.PostForm(url, data)

		if err != nil{
			logger.Do.Println("Post Requests Error happens retry ..... ,", err)
			time.Sleep(time.Second*4)
			NTimes--
		}else{
			break
		}
	}

	var res map[string]interface{}

	_ = json.NewDecoder(resp.Body).Decode(&res)

	return nil
}

func PostJson(url string, js string) {
	/*
		eg:
			url: "http://localhost:9089/submit"
			var js string =`{"Name":"naili","Age":123}`
	*/
	url = strings.TrimSpace(url)
	var jsonStr = []byte(js)
	req, err := http.NewRequest(http.MethodPost, url, bytes.NewBuffer(jsonStr))
	if err != nil {
		panic("connection errro")
	}
	req.Header.Set("Content-Type", "application/json")

	client := &http.Client{}
	resp, err := client.Do(req)
	if err != nil {
		panic(err)
	}
	defer resp.Body.Close()

	logger.Do.Println("response Status:", resp.Status)
	logger.Do.Println("response Headers:", resp.Header)
	body, _ := ioutil.ReadAll(resp.Body)
	logger.Do.Println("response Body:", string(body))
}
