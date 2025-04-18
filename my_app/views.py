import os
from django.shortcuts import render
from django.conf import settings
from .predict import predict_image  

def home(request):
    if request.method == 'POST' and request.FILES['image']:
        image = request.FILES['image']
        image_name = image.name  # Just get the image name, not the full path
        
        # Save the image to the media folder
        image_path = os.path.join(settings.MEDIA_ROOT, image_name)
        with open(image_path, 'wb') as f:
            f.write(image.read())
        
        # Predict the image class
        predicted_class, confidence = predict_image(image_path)

        # Construct the correct image URL
        image_url = settings.MEDIA_URL + image_name

        return render(request, 'my_app/home.html', {
            'predicted_class': predicted_class,
            'confidence': confidence,
            'image_url': image_url,  # Use the image URL directly for the template
        })

    return render(request, 'my_app/home.html')
