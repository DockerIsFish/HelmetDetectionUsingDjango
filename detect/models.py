from django.db import models

class registeration(models.Model):
    name = models.CharField(max_length=200)
    id_number = models.TextField()
    datetime = models.DateTimeField()

    def __str__(self):
        return self.subject
