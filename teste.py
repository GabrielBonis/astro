import requests

url = "http://astro.gbonis.com.br/sky-map"
dados = {
  "lat": -23.55,
  "lon": -46.63,
  "date": "2022-01-08",
  "title": "O NASCER DE UMA ESTRELA"
}

print("Enviando requisição para a API...")

try:
    response = requests.post(url, json=dados)

    if response.headers.get('content-type') == 'image/png':
        with open("meu_quadro_estelar.png", "wb") as f:
            f.write(response.content)
        print("✨ SUCESSO ABSOLUTO! A imagem foi salva como 'meu_quadro_estelar.png'. Vá dar uma olhada!")
    else:
        print("🚨 A API retornou um ERRO. Veja os detalhes abaixo:")
        print(response.json())

except requests.exceptions.ConnectionError:
    print("🚨 ERRO DE CONEXÃO: Você esqueceu de rodar o servidor Uvicorn?")