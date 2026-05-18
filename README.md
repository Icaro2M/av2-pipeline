# 🏠 House Prices Prediction - Starter Kit

Bem-vindo ao repositório base para o desafio de predição de preços de imóveis! Este kit foi projetado para ajudá-lo a organizar sua submissão e garantir que seu modelo seja avaliado corretamente pelo nosso sistema automático.

## 🚀 Como usar este template

1.  **Clone o Repositório:** Comece fazendo o clone ou fork deste repositório para o seu ambiente de desenvolvimento.
2.  **Treine seu Modelo:** Utilize seus notebooks ou scripts para explorar os dados e treinar seu melhor modelo.
3.  **Substitua o Baseline:** Salve seu modelo final na pasta raiz. Você pode substituir o arquivo `modelo_baseline.joblib` pelo seu próprio arquivo (seja `.joblib`, `.pkl`, etc.).
4.  **Atualize o `pipeline.py`:** Se você alterou o nome ou o formato do arquivo do modelo, lembre-se de atualizar o caminho dentro da função `prever_precos`.
5.  **Teste Localmente:** Antes de enviar, execute o comando abaixo no terminal para garantir que tudo está funcionando:
    ```bash
    python pipeline.py
    ```

## ⚠️ Regras Cruciais (Não ignore!)

> **IMPORTANTE:** Você **NÃO PODE** alterar o nome da função `prever_precos` nem o seu parâmetro de entrada (`caminho_arquivo_teste`) no arquivo `pipeline.py`. O corretor automático chama exatamente esta função. Alterar sua assinatura resultará em erro na correção e nota zero na avaliação automatizada.

## 📦 Dependências

Sempre que utilizar uma nova biblioteca (como `xgboost`, `scipy`, etc.), certifique-se de adicioná-la ao arquivo `requirements.txt`. O ambiente de correção instalará as dependências listadas lá.

---
Boa sorte com o desenvolvimento do seu modelo! 🚀
