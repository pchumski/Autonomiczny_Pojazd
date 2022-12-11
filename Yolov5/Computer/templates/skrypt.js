function zmien(){
    var btn = document.getElementsByName("in").innerHTML;
    if (btn == "START CAMERA"){
        document.getElementsByName("przycisk").innerHTML =  "STOP CAMERA";
    }
    else{
        document.getElementsByName("przycisk").innerHTML = "START CAMERA";
    }
}