$(document).ready(function() {
    // ========== PASSWORD PROTECTION ==========
    const SITE_PASSWORD = 'autonomous';
    
    // Show main content and hide password overlay
    function showMainContent() {
        $('#password-overlay').fadeOut(300, function() {
            $(this).remove();
        });
        $('#main-content').fadeIn(300);
    }
    
    // Handle password submission
    function handlePasswordSubmit() {
        const password = $('#password-input').val().trim();
        const errorDiv = $('#password-error');
        
        if (password === SITE_PASSWORD) {
            // Correct password
            errorDiv.addClass('is-hidden');
            $('#password-input').val('');
            showMainContent();
        } else {
            // Incorrect password
            errorDiv.removeClass('is-hidden');
            $('#password-input').val('');
            $('#password-input').focus();
            // Shake animation
            $('#password-input').addClass('is-danger');
            setTimeout(function() {
                $('#password-input').removeClass('is-danger');
            }, 500);
        }
    }
    
    // Set up password form handlers
    $('#password-submit').on('click', function(e) {
        e.preventDefault();
        handlePasswordSubmit();
    });
    
    $('#password-input').on('keypress', function(e) {
        if (e.which === 13) { // Enter key
            e.preventDefault();
            handlePasswordSubmit();
        }
    });
    
    // Password visibility toggle
    $(document).on('click', '#password-toggle', function(e) {
        e.preventDefault();
        e.stopPropagation();
        const input = $('#password-input');
        const icon = $('#password-toggle-icon');
        
        if (input.attr('type') === 'password') {
            input.attr('type', 'text');
            icon.removeClass('fa-eye').addClass('fa-eye-slash');
        } else {
            input.attr('type', 'password');
            icon.removeClass('fa-eye-slash').addClass('fa-eye');
        }
        return false;
    });
    
    // Also handle click on the icon itself
    $(document).on('click', '#password-toggle-icon', function(e) {
        e.preventDefault();
        e.stopPropagation();
        $('#password-toggle').click();
        return false;
    });
    
    // Always show password prompt on page load (no session storage)
    // Focus on password input
    setTimeout(function() {
        $('#password-input').focus();
    }, 100);
    
    // Check for click events on the navbar burger icon
    $(".navbar-burger").click(function() {
      // Toggle the "is-active" class on both the "navbar-burger" and the "navbar-menu"
      $(".navbar-burger").toggleClass("is-active");
      $(".navbar-menu").toggleClass("is-active");
    });

    var options = {
			slidesToScroll: 1,
			slidesToShow: 3,
			loop: true,
			infinite: true,
			autoplay: false,
			autoplaySpeed: 3000,
    }

		// Initialize all div with carousel class
    var carousels = bulmaCarousel.attach('.carousel', options);

    // Loop on each carousel initialized
    for(var i = 0; i < carousels.length; i++) {
    	// Add listener to  event
    	carousels[i].on('before:show', state => {
    		console.log(state);
    	});
    }
    
})
