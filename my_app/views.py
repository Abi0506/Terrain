import os
from django.shortcuts import render
from django.conf import settings
from .predict import predict_image  

def home(request):
    if request.method == 'POST' and request.FILES['image']:
        image = request.FILES['image']
        image_name = image.name  
        
        
        image_path = os.path.join(settings.MEDIA_ROOT, image_name)
        with open(image_path, 'wb') as f:
            f.write(image.read())
        
       
        predicted_class, confidence = predict_image(image_path)

       
        image_url = settings.MEDIA_URL + image_name

        return render(request, 'my_app/home.html', {
            'predicted_class': predicted_class,
            'confidence': confidence,
            'image_url': image_url,  
        })

    return render(request, 'my_app/home.html')
