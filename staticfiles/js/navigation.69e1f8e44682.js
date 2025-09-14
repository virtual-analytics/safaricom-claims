document.addEventListener('DOMContentLoaded', function() {
    // Handle submenu toggles
    const submenuToggles = document.querySelectorAll('.submenu-toggle');
    
    submenuToggles.forEach(toggle => {
        toggle.addEventListener('click', function(e) {
            e.preventDefault();
            const parent = this.parentElement.parentElement;
            const submenu = parent.querySelector('.submenu');
            
            if (parent.classList.contains('open')) {
                parent.classList.remove('open');
                submenu.style.display = 'none';
            } else {
                parent.classList.add('open');
                submenu.style.display = 'flex';
            }
        });
    });

    // Auto-expand active submenu
    const activeSubmenu = document.querySelector('.submenu li.selected');
    if (activeSubmenu) {
        const parent = activeSubmenu.closest('.has-submenu');
        if (parent) {
            parent.classList.add('open');
            const submenu = parent.querySelector('.submenu');
            if (submenu) {
                submenu.style.display = 'flex';
            }
        }
    }

    // Handle parent link clicks
    const hasSubmenuLinks = document.querySelectorAll('.has-submenu > a');
    hasSubmenuLinks.forEach(link => {
        link.addEventListener('click', function(e) {
            if (e.target.classList.contains('submenu-toggle')) {
                return;
            }
            
            const parent = this.parentElement;
            const submenu = parent.querySelector('.submenu');
            
            if (parent.classList.contains('open')) {
                parent.classList.remove('open');
                submenu.style.display = 'none';
            } else {
                parent.classList.add('open');
                submenu.style.display = 'flex';
            }
        });
    });
});
