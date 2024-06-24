document.getElementById('send').addEventListener('click', uploadImage);

function uploadImage() {
    const fileInput = document.getElementById('file-input');
    const file = fileInput.files[0];
    
    if (!file) {
        alert("Por favor, selecciona una imagen primero.");
        return;
    }

    const formData = new FormData();
    formData.append('file', file);

    fetch('http://localhost:5000/predict', {
        method: 'POST',
        body: formData
    })
    .then(response => response.json())
    .then(data => {
        if (data.result) {
            alert(`Resultado del servidor: ${data.result}`);
        } else {
            alert('Respuesta del servidor: ' + JSON.stringify(data));
        }
    })
    .catch((error) => {
        console.error('Error:', error);
        alert('Error al subir la imagen');
    });
}
