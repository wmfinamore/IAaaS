import h2o
import pandas as pd

from rest_framework import generics
from rest_framework.response import Response

from .models import ProcessamentoModeloMachineLearning, ModeloMachineLearningProcessado
from .serializers import ProcessamentoModeloMachineLearningCreateSerializer, PrevisaoSerializer


# Herda CreateAPIView de rest_framework que esta em generics
class ProcessamentoModeloMachineLearningView(generics.CreateAPIView):
    # importamos o serializador criado
    serializer_class = ProcessamentoModeloMachineLearningCreateSerializer

    # função para salvar, sobrescreve o método da classe CreateApiView
    def perform_create(self, serializer):
        # salva o modelo e retorna código 200
        q = serializer.save()
        # no nosso caso além de gravar os dados precisamos processar - treinar o modelo
        # gera dados do modelos processados
        q.processar()


class PrevisaoView(generics.views.APIView):
    serializer_class = PrevisaoSerializer

    # sobrescreve método post de ApiView
    def post(self, request):
        try:
            # Recebe os dados enviados pela requisição
            model_id = request.POST.get('model_id')
            csv_prever = request.FILES['csv_prever']

            if model_id:
                # Busca o modelo usando o ORM do Django pelo model_id
                modelo_processado = ModeloMachineLearningProcessado.objects.get(model_id=model_id)

                # Busca o processamento vinculado ao modelo processado usando a ORM do Django
                processamento = modelo_processado.processamentos.first()
            else:
                # se não informou o model_id, busca-se o melhor modelo do processamento mais recente
                processamento = ProcessamentoModeloMachineLearning.objects.all().first()  # o processamento mais recente
                modelo_processado = processamento.modelos_processados.all().first()
                # o primeiro modelo sempre é o melhor

            # faz a leitura do arquivo para previsão fazendo uso da biblioteca Pandas
            teste = pd.read_csv(csv_prever, sep=";")
            colunas_enviadas = ','.join(teste.columns.tolist())
            if processamento.variaveis_independentes != colunas_enviadas:
                raise Exception('Erro no layout do arquivo de previsão: Para este modelo são essenciais as seguintes '
                                'colunas:"{variaveis_independentes}", mas você enviou as colunas: "{colunas_enviadas}"'
                                .format(variaveis_independentes=processamento.variaveis_independentes,
                                        colunas_enviadas=colunas_enviadas))

            # Inicializa a conexão com o h2o
            h2o.init()
            teste = h2o.H2OFrame(teste)

            # Fazer o load do binário do modelo
            modelo_automl = h2o.load_model(modelo_processado.binario_modelo.name)
            prever = modelo_automl.predict(teste)

            data_frame = prever.as_data_frame()

            """
                Formatar os dados de uma forma mais simples, para percorrer depois no JavaScript
            """
            previsoes = list()

            for i in range(0, len(data_frame['predict'])-1):
                previsoes.append({
                    'predict': data_frame['predict'][i],
                    'p0': data_frame['p0'][i],
                    'p1': data_frame['p1'][i]
                })

            return Response(status=201, data={'previsoes': previsoes})
        except Exception as e:
            return Response(status=401, data={'Erro': str(e)})
