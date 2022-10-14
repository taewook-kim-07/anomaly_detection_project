document.write('<script src="https://cdn.jsdelivr.net/gh/ericnograles/browser-image-resizer@2.4.0/dist/index.js"></script>');
const resize_config = {
    quality: 1.0,
    maxWidth: 512,
    maxHeight: 512
};

const modelInput = document.getElementById('select_model');
const fileInput = document.getElementById('file');
const predictButton = document.getElementById("predict");
const clearButton = document.getElementById("clear");
const preview = document.getElementById("preview");
print_index = 0

const predict = async (select_model) => {
    var headers = new Headers();
    var csrftoken = getCookieJS('csrftoken');
    headers.append('X-CSRFToken', csrftoken);

    const files = fileInput.files;

    print_index += 1
    preview.innerHTML += `<div id="print_row-${print_index}" class="row row-cols-2 mt-2"></div><hr class="mt-0 mb-0">`;
    const print_row = document.getElementById(`print_row-${print_index}`);

    [...files].map(async (img) => {
        let resizedImage = await BrowserImageResizer.readAndCompressImage(img, resize_config);
        const data = new FormData();
        data.append('select_model', select_model);
        data.append('file', resizedImage);
        const result = await fetch("/api/predict/",
            {
                method: 'POST',
                headers: headers,
                credentials: 'include',
                body: data,
            }).then(response => {
                return response.json();
            }).catch((error) => {
                return 'ERROR';
            });
        renderImageLabel(select_model, img.name, resizedImage, result.predict, print_row);
    })
};

const renderImageLabel = (select_model, filename, img, label, print_row) => {
    const reader = new FileReader();
    reader.onload = () => {
        badge = ( String(label).replace('_', '-') == filename.trim().replace(/(.png|.jpg|.jpeg)$/,'').replace('_', '-') )?'<span class="badge bg-success">TRUE</span>':'<span class="badge bg-danger">FALSE</span>';
        print_row.innerHTML += `
      <div class="col-2 mb-0">
        <div class="card text-center">
          <img src="${reader.result}" class="card-img-top mx-auto d-block" style="max-width: 250px; height: auto;">
          <div class="card-body">
            <h5 class="card-title">${label}<br>${badge}</h5>
            <p class="card-text">${select_model}<br>${filename}</p>
          </div>
        </div>
    </div>`;

    };
    reader.readAsDataURL(img);
};

const getCookieJS = (cname) => {
    var name = cname + "=";
    var decodedCookie = decodeURIComponent(document.cookie);
    var ca = decodedCookie.split(';');
    for(var i = 0; i <ca.length; i++) {
        var c = ca[i];
        while (c.charAt(0) == ' ') {
            c = c.substring(1);
        }
        if (c.indexOf(name) == 0) {
            return c.substring(name.length, c.length);
        }
    }
    return "";
}

predictButton.addEventListener("click", () => predict(modelInput.value));
clearButton.addEventListener("click", () => preview.innerHTML = "");