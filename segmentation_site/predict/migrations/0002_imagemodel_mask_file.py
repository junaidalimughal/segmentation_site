# Generated by Django 4.0.3 on 2022-03-20 17:06

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('predict', '0001_initial'),
    ]

    operations = [
        migrations.AddField(
            model_name='imagemodel',
            name='mask_file',
            field=models.FileField(null=True, upload_to=''),
        ),
    ]
