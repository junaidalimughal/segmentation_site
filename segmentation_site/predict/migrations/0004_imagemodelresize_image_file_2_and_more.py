# Generated by Django 4.0.3 on 2022-03-22 19:25

from django.db import migrations, models
import django_resized.forms


class Migration(migrations.Migration):

    dependencies = [
        ('predict', '0003_imagemodelresize'),
    ]

    operations = [
        migrations.AddField(
            model_name='imagemodelresize',
            name='image_file_2',
            field=django_resized.forms.ResizedImageField(crop=None, force_format='JPEG', keep_meta=True, null=True, quality=75, size=[512, 512], upload_to='media'),
        ),
        migrations.AddField(
            model_name='imagemodelresize',
            name='mask_file_2',
            field=models.FileField(null=True, upload_to=''),
        ),
    ]