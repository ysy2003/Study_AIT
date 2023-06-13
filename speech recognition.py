import sys
import requests
import json
import base64
import os
import logging
import speech_recognition as sr

def get_token():
    #Get token
    baidu_server = "https://openapi.baidu.com/oauth/2.0/token?"
    grant_type = "client_credentials"
    client_id = "U7YIXKu0B2wsGjBu1cgaa0GF"
    client_secret = "rFdqMLA1NrTuFGiCzw9tn9WuU6uHOxUf" # YU SHUYANG's ID and keys, can change and use another one

    #get url
    url = f"{baidu_server}grant_type={grant_type}&client_id={client_id}&client_secret={client_secret}"
    res = requests.post(url)
    token = json.loads(res.text)["access_token"]
    return token


def audio_baidu(filename):
    with open(filename, "rb") as f:
        speech = base64.b64encode(f.read()).decode('utf-8')
    size = os.path.getsize(filename)
    token = get_token()
    headers = {'Content-Type': 'application/json'}
    url = "https://vop.baidu.com/server_api"
    data = {
        "format": "wav",
        "rate": "16000",
        "dev_pid": "1736",
        "speech": speech,
        "cuid": "TEDxPY",
        "len": size,
        "channel": 1,
        "token": token,
    }

    req = requests.post(url, json.dumps(data), headers)
    result = json.loads(req.text)

    if result["err_msg"] == "success.":
        print(result['result'])
        return result['result']
    else:
        print("Content acquisition failure, exit speech recognition")
        return -1

def main():
    logging.basicConfig(level=logging.INFO)
    wav_num = 0
    while True:
        r = sr.Recognizer()
        #Enable microphone
        mic = sr.Microphone()
        logging.info('In the recording...')
        with mic as source:
            #Noise reduction
            r.adjust_for_ambient_noise(source)
            audio = r.listen(source)
        with open(f"00{wav_num}.wav", "wb") as f:
            #Save the recorded sound from the microphone as a wav file
            f.write(audio.get_wav_data(convert_rate=16000))
        logging.info('End of recording, identification in progress...')
        target = audio_baidu(f"00{wav_num}.wav")
        if target == -1:
            break
        wav_num += 1
        if target == ['Stop。']:
            sys.exit()

main()