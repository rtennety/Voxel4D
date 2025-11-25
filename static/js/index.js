$(document).ready(function() {
    // ========== PASSWORD PROTECTION ==========
    const SITE_PASSWORD = 'autonomous4d';
    
    // Show main content and hide password overlay
    function showMainContent() {
        $('#password-overlay').fadeOut(300, function() {
            $(this).remove();
        });
        $('#main-content').fadeIn(300);
    }
    
    // Handle password submission
    function handlePasswordSubmit() {
        const password = $('#password-input').val();
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
    
    // ========== IMAGE PROTECTION ==========
    
    // Wrap all images in protective containers
    $('img').each(function() {
        var $img = $(this);
        var $wrapper = $('<div class="image-protected"></div>');
        var $overlay = $('<div class="image-overlay"></div>');
        
        $img.wrap($wrapper);
        $img.after($overlay);
    });
    
    // Disable right-click on images and overlays
    $(document).on('contextmenu', 'img, .image-overlay, .image-protected', function(e) {
        e.preventDefault();
        e.stopPropagation();
        return false;
    });
    
    // Prevent drag on images
    $(document).on('dragstart', 'img', function(e) {
        e.preventDefault();
        return false;
    });
    
    // Prevent image selection
    $(document).on('selectstart', 'img', function(e) {
        e.preventDefault();
        return false;
    });
    
    // Block common image saving keyboard shortcuts
    $(document).on('keydown', function(e) {
        // Block Ctrl+S (Save)
        if (e.ctrlKey && e.key === 's') {
            e.preventDefault();
            return false;
        }
        // Block Ctrl+Shift+I (Inspect - might reveal image URLs)
        if (e.ctrlKey && e.shiftKey && e.key === 'I') {
            e.preventDefault();
            return false;
        }
    });
    
    // Prevent opening image in new tab
    $(document).on('click', 'img', function(e) {
        e.preventDefault();
        return false;
    });
    
    // Block image loading via right-click menu (additional protection)
    document.addEventListener('DOMContentLoaded', function() {
        var images = document.querySelectorAll('img');
        images.forEach(function(img) {
            // Remove ability to open image in new tab
            img.setAttribute('oncontextmenu', 'return false;');
            img.setAttribute('ondragstart', 'return false;');
            img.setAttribute('onselectstart', 'return false;');
        });
    });
})
