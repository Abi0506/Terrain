import os
from django.conf import settings
from django.core.files.storage import default_storage
from django.shortcuts import render
from .predict import predict_image  # Your model inference logic

def home(request):
    if request.method == 'POST' and request.FILES.get('image'):
        image_file = request.FILES['image']
        
        # Save the image in the media/ folder
        filename = default_storage.save(f'{image_file.name}', image_file)
        file_path = os.path.join(settings.MEDIA_ROOT, filename)

        # Now pass the real file path to the model
        prediction = predict_image(file_path)

        return render(request, 'my_app/home.html', {
            'prediction': prediction,
            'uploaded': True,
            'image_url': settings.MEDIA_URL + filename
        })

    return render(request, 'my_app/home.html')
