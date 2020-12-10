// client-side js
// run by the browser each time your view template is loaded

// by default, you've got jQuery,
// add other scripts at the bottom of index.html


$(function() {
  
  $("form").submit(function(event) {
    event.preventDefault();
    let form_data = $('form').serializeArray();

    console.log(form_data);
    $.post("/recommend?" + $.param({ form_data: form_data }), function(data) {
      console.log(data);
    });
  });
  
});
