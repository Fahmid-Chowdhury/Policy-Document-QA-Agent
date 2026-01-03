from rest_framework import serializers


class IndexRequest(serializers.Serializer):
    docs_path = serializers.CharField(required=True)
    rebuild = serializers.BooleanField(default=False)
    embedding = serializers.ChoiceField(choices=["google", "hf"], default="google")


class AskRequest(serializers.Serializer):
    question = serializers.CharField(required=True, max_length=800)
    k = serializers.IntegerField(default=6, min_value=1, max_value=50)
    mmr = serializers.BooleanField(default=False)
    fetch_k = serializers.IntegerField(default=30, min_value=1, max_value=200)
    embedding = serializers.ChoiceField(choices=["google", "hf"], default="google")
    llm_model = serializers.ChoiceField(choices=["google", "hf"], default="google")
    docs_path = serializers.CharField(required=False, allow_null=True, allow_blank=True)


class AskJsonRequest(AskRequest):
    out = serializers.CharField(required=False, allow_null=True, allow_blank=True)
