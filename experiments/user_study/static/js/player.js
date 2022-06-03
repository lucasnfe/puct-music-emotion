let pieceEnded = false;

var music = document.getElementById('pieceAudio'); // id for audio element
var duration = music.duration; // Duration of audio clip, calculated here for embedding purposes
var pButton = document.getElementById('pButton'); // play button
var pButtonImg = document.getElementById('pButtonImg'); // play button image
var playhead = document.getElementById('playhead'); // playhead
var timeline = document.getElementById('timeline'); // timeline

// timeline width adjusted for playhead
var timelineWidth = timeline.offsetWidth - playhead.offsetWidth;

// timeupdate event listener
music.addEventListener("timeupdate", timeUpdate, false);

// makes timeline clickable
timeline.addEventListener("click", function (event) {
      // Only allow piece drag after finishing for the first time
      if(!pieceEnded) {
        return;
      }

     moveplayhead(event);
     music.currentTime = duration * clickPercent(event);
}, false);

// returns click as decimal (.77) of the total timelineWidth
function clickPercent(event) {
     return (event.clientX - getPosition(timeline)) / timelineWidth;
}

// makes playhead draggable
playhead.addEventListener('mousedown', mouseDown, false);
window.addEventListener('mouseup', mouseUp, false);

// Boolean value so that audio position is updated only when the playhead is released
var onplayhead = false;

// mouseDown EventListener
function mouseDown() {
      // Only allow piece drag after finishing for the first time
      if(!pieceEnded) {
        return;
      }

     onplayhead = true;
     window.addEventListener('mousemove', moveplayhead, true);
     music.removeEventListener('timeupdate', timeUpdate, false);
}

// mouseUp EventListener
// getting input from all mouse clicks
function mouseUp(event) {
    // Only allow piece drag after finishing for the first time
    if(!pieceEnded) {
      return;
    }

     if (onplayhead == true) {
         moveplayhead(event);
         window.removeEventListener('mousemove', moveplayhead, true);
         // change current time
         music.currentTime = duration * clickPercent(event);
         music.addEventListener('timeupdate', timeUpdate, false);
     }
     onplayhead = false;
}

// mousemove EventListener
// Moves playhead as user drags
function moveplayhead(event) {
      // Only allow piece drag after finishing for the first time
      if(!pieceEnded) {
        return;
      }

     var newMargLeft = event.clientX - getPosition(timeline);

     if (newMargLeft >= 0 && newMargLeft <= timelineWidth) {
         playhead.style.marginLeft = newMargLeft + "px";
     }
     if (newMargLeft < 0) {
         playhead.style.marginLeft = "0px";
     }
     if (newMargLeft > timelineWidth) {
         playhead.style.marginLeft = timelineWidth + "px";
     }
}

// timeUpdate
// Synchronizes playhead position with current point in audio
function timeUpdate() {
     var playPercent = timelineWidth * (music.currentTime / duration);
     playhead.style.marginLeft = playPercent + "px";
     if (music.currentTime == duration) {
         pButton.className = "";
         pButton.className = "fas fa-play";
     }
}

// Gets audio file duration
music.addEventListener("canplaythrough", function () {
     duration = music.duration;
}, false);

// getPosition
// Returns elements left position relative to top-left of viewport
function getPosition(el) {
     return el.getBoundingClientRect().left;
}
