import qrcode

url = "https://akash-chatbot-fvalyflxnqmkpc3pq3qbbh.streamlit.app/"
img = qrcode.make(url)
img.save("akash_chatbot_qr.png")
