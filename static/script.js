var swiper = new Swiper('.blog-slider', {
      spaceBetween: 30,
      effect: 'fade',
      loop: true,
      // mousewheel: {
      //   invert: false,
      // },
      // autoHeight: true,
      pagination: {
        el: '.blog-slider__pagination',
        clickable: true,
      }
    });

// function playAudio() {
//   new Audio("{{ url_for('static', filename='song.mp3') }}").play();
// }    