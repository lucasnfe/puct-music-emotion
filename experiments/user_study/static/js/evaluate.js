// Define sound instance
let sound = null;

// Controls
let transPos = 0;
let thumbPos = 0;

let didClickOnThumb = false;
let didClickOnTrans = false;

// Keep track of how many times the user pressed play
let playCount = 0;
let playCompleteCount = 0;

const slider = $("#emotion-slider");
const thumb  = $("#emotion-slider-thumb");
const trans  = $("#emotion-trans-slider-thumb");

const transTimer = $("#emotion-trans-timer");
const durationTimer = $("#duration");
const elapsedTimer = $("#timer");

const leftTrack = $('#emotion-track-left');
const rightTrack = $('#emotion-track-right');

const qualityLikert = $(".evaluate-quality");

const emotion1Select = $('#emotion1');
const emotion2Select = $('#emotion2');

thumb.on("mousedown", function(ev) {
    thumbDown();
});

thumb.on("touchstart", function(ev) {
    thumbDown();
});

trans.on("mousedown", function(ev) {
    transDown();
});

trans.on("touchstart", function(ev) {
    transDown();
});

$(document).on("mousemove", function(ev) {
    thumbMove(ev.clientX);
    transMove(ev.clientX);
});

$(document).on("touchmove", function(ev) {
    if(didClickOnThumb || didClickOnTrans) {
        ev.preventDefault();
    }

    var touches = ev.changedTouches;
    for (let i = 0; i < touches.length; i++) {
        thumbMove(touches[i].pageX);
        transMove(touches[i].pageX);
    }
});

$(document).on("mouseup", function(ev) {
    thumbUp();
    transUp();
});

$(document).on("touchend", function(ev) {
    thumbUp();
    transUp();
});

$(document).on("touchcancel", function(ev) {
    thumbUp();
    transUp();
});

qualityLikert.on('click', function(ev){
    if(playCompleteCount == 0) {
        ev.preventDefault();

        // Pause the piece
        pause();

        // Show play popover
        $("#playButton").popover('show');
    }
});

leftTrack.on("mousedown", function(ev) {
    trackDown(ev);
});

rightTrack.on("mousedown", function(ev) {
    trackDown(ev);
});

emotion1Select.on('hide.bs.select', function () {
    if(playCompleteCount == 0) {
        $(this).val("");
        $(this).change();

        // Pause the piece
        pause();

        // Show play popover
        $("#playButton").popover('show');
    }
    else {
        onEmotionButtonPlayed();
    }
});

emotion1Select.on('show.bs.select', function () {
    $(this).popover('hide');
});

emotion2Select.on('hide.bs.select', function () {
    if(playCompleteCount == 0) {
        $(this).val("");
        $(this).change();

        // Pause the piece
        pause();

        // Show play popover
        $("#playButton").popover('show');
    }
    else {
        onEmotionButtonPlayed();
    }
});

emotion2Select.on('show.bs.select', function () {
    $(this).popover('hide');
});

function transDown() {
    $("#emotion-trans-slider-thumb").popover('hide');

    if(sound.state() != "loaded") {
        return;
    }

    didClickOnTrans = true;

    // Hide trans timer
    transTimer.css("visibility", "visible");
}

function thumbDown() {
    if(sound == null || sound.state() != "loaded") {
        return;
    }

    didClickOnThumb = true;
    pause();
}

function transMove(clientX) {
    if(sound == null || sound.state() != "loaded") {
        return;
    }

    if(didClickOnTrans && playCompleteCount > 0) {
        // Calculate mid point
        let p = calcMidPoint(clientX, trans.width(), transLimit);

        // Update thumb position
        updateTrans(p);
    }
}

function thumbMove(clientX) {
    if(sound == null || sound.state() != "loaded") {
        return;
    }

    if(didClickOnThumb) {
        // Calculate mid point
      	let p = calcMidPoint(clientX, thumb.width(), thumbLimit);

        // Update thumb position
        updateProgressBar(p);
    }
}

function transUp() {
    if(sound == null || sound.state() != "loaded") {
        return;
    }

    didClickOnTrans = false;

    // Hide trans timer
    transTimer.css("visibility", "hidden");
}

function thumbUp() {
    if(sound == null || sound.state() != "loaded") {
        return;
    }

    didClickOnThumb = false;
}

function trackDown(ev) {
    if(sound == null || sound.state() != "loaded") {
        return;
    }

    if(!didClickOnThumb && playCompleteCount > 0) {
        pause();

        // Calculate mid point
        let p = calcMidPoint(ev.clientX, thumb.width(), thumbLimit);

        // Update thumb position
        updateProgressBar(p);
    }
}

function switchSelect(select, track) {
    switch (select.value) {
        default:
            track.css("backgroundColor", "#616A71");
            break;
        case "1":
            track.css("backgroundColor", "#92C1E9");
            break;
        case "2":
            track.css("backgroundColor", "#F3DD6D");
            break;
        case "3":
            track.css("backgroundColor", "#FF8C00");
            break;
        case "4":
            track.css("backgroundColor", "#C3B1A4");
            break;
    }
}

function play() {
    if(!sound.playing() && sound.seek() == sound.duration()) {
        updateProgressBar(0);
    }

    setPlayImg();
    sound.play();
}

function pause() {
    setPauseImg();
    sound.pause();
}

function step() {
    // Determine our current seek position.
    let p = sound.seek()/sound.duration();

    updateProgressBar(p);

    if(playCompleteCount == 0) {
        updateEmotionSlider(p);
    }

    // If the sound is still playing, continue stepping.
    if (sound.playing()) {
        requestAnimationFrame(step);
    }
}

function onPlayButtonPlayed() {
    $("#playButton").popover('hide');

    if(!sound.playing()) {
        play();
    }
    else {
        pause();
    }
}

function onEmotionButtonPlayed() {
    switchSelect(emotion1, leftTrack);
    switchSelect(emotion2, rightTrack);

    pause();
}

function updateEmotionSlider(p) {
    leftTrack.css("width", (p * 100) + "%");
    rightTrack.css("width", ((1.0 - p) * 100) + "%");
}

// Updates
function updateTimer(timerElement, p) {
    if(p >= 0 && p <= 1.0) {
        let time = formatTime(Math.round(p * sound.duration()));
        timerElement.html(time);
    }
}

function updateTrans(p) {
    // Compute seek position
    transPos = clamp(p, 0.0, 1.0);

    // Update track widths
    updateEmotionSlider(transPos);

    // Update track colors
    switchSelect(emotion1, leftTrack);
    switchSelect(emotion2, rightTrack);

    // Update trans position
    setTransPosition(transPos);

    // Update trans timer
    updateTimer(transTimer, transPos);

    trans.css("visibility", "visible");
}

function updateProgressBar(p) {
    // Compute seek position
    thumbPos = clamp(p, 0.0, 1.0);
    let seek = thumbPos * sound.duration();

    // Seek sound
    if(!sound.playing()) {
        sound.seek(seek);
    }

    // Update thumb position
    setThumbPosition(thumbPos);

    if(playCompleteCount == 0) {
        thumb.css("visibility", "hidden");
    }
    else {
        thumb.css("visibility", "visible");
    }

    // Update timer
    updateTimer(elapsedTimer, thumbPos);
}

function calcMidPoint(x, width, limit) {
    return (x - limitX0)/limit;
}

function setThumbPosition(p) {
    thumb.css("left", (limitX1 + p * thumbLimit - thumb.width()/2) + "px");
}

function setTransPosition(p) {
    transTimer.css("left", (limitX1 + p * transLimit - transTimer.width()/2) + "px");
    trans.css("left", (limitX1 + p * transLimit - trans.width()/2) + "px");
}

function onResize() {
    sliderPaddingLeft = parseInt(slider.css("padding-left").split("px")[0]);

    thumbLimit = slider.width() - thumb.width()/2;
    transLimit = slider.width() - trans.width()/2;

    // Compute slider limits
    limitX0 = slider.offset().left + sliderPaddingLeft;
    limitY0 = slider.offset().top;

    // Compute slider init position
    limitX1 = slider.position().left + sliderPaddingLeft;

    setThumbPosition(thumbPos);
    setTransPosition(transPos);

    thumb.draggable({axis: "x", containment: [limitX0 - thumb.width()/2, limitY0,
                                              limitX0 + thumbLimit, limitY0]});

    trans.draggable({axis: "x", containment: [limitX0 - trans.width()/2, limitY0,
                                              limitX0 + transLimit, limitY0]});
}

// Formatting
function formatTime(secs) {
  var minutes = Math.floor(secs / 60) || 0;
  var seconds = (secs - minutes * 60) || 0;

  return minutes + ':' + (seconds < 10 ? '0' : '') + seconds;
}

function clamp(num, min, max) {
  return num <= min ? min : num >= max ? max : num;
}
