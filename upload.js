//upload.js
(function() {
    $('#Upload').click(function(){
        var fd = new FormData($('#file')[0]);
        fd.stopPropogation();
        $.ajax({
            type: 'POST',
            url: '/success',
            data: form_data,
            contentType: false,
            cache: false,
            processData: false,
            success: function(data) {
                console.log('Success!');
            },
        });
    });
});