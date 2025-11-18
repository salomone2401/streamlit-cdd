import google.generativeai as genai

genai.configure(api_key="AIzaSyDCquiuvOh-qdRxd9oYpIEY75Zf_TzeJ9Q")

model = genai.GenerativeModel("gemini-pro")
response = model.generate_content("Dime un chiste sobre pandas (los animales)")
print(response)
print("Texto:", response.text)