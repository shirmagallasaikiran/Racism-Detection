from django.db import models

# Create your models here.
from django.db.models import CASCADE


class ClientRegister_Model(models.Model):
    id = models.BigAutoField(primary_key=True)
    username = models.CharField(max_length=30)
    email = models.EmailField(max_length=30)
    password = models.CharField(max_length=10)
    phoneno = models.CharField(max_length=10)
    country = models.CharField(max_length=30)
    state = models.CharField(max_length=30)
    city = models.CharField(max_length=30)
    gender= models.CharField(max_length=30)
    address= models.CharField(max_length=30)


class sentiment_analysis_type(models.Model):
    id = models.BigAutoField(primary_key=True)

    tid= models.CharField(max_length=300)
    tweet= models.CharField(max_length=30000)
    Prediction= models.CharField(max_length=300)

class detection_accuracy(models.Model):
    id = models.BigAutoField(primary_key=True)

    names = models.CharField(max_length=300)
    ratio = models.CharField(max_length=300)

class detection_ratio(models.Model):
    id = models.BigAutoField(primary_key=True)

    names = models.CharField(max_length=300)
    ratio = models.CharField(max_length=300)



