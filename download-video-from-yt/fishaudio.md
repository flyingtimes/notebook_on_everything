from fish_audio_sdk import Session, TTSRequest, ReferenceAudio

session = Session("your_api_key")

with open("output1.mp3", "wb") as f:
    for chunk in session.tts(TTSRequest(
        reference_id="MODEL_ID_UPLOADED_OR_CHOSEN_FROM_PLAYGROUND",
        text="Hello, world!"
    )):
        f.write(chunk)