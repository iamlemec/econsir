//
// fixed elements
//

var slide_act = $('#pol-act');
var slide_cut = $('#pol-cut');
var slide_lag = $('#pol-lag');

var check_tcase = $('#opt-tcase');
var check_kzone = $('#opt-kzone');
var check_vax = $('#opt-vax');

var prof_smash = $('#prof-smash');
var prof_erad = $('#prof-erad');
var prof_flat = $('#prof-flat');
var prof_lfaire = $('#prof-lfaire');
var prof_vax = $('#prof-vax');

var chart_box = $('#chart-box');

var popup = $('#popup');

//
// render policies
//

var conf = {'actions': false, 'renderer': 'svg'};

// render policy graphs
function renderPolicy(act, cut, lag, tcase, kzone, vax) {
    console.log(act, cut, lag, tcase, kzone, vax);

    chart_box.fadeTo('fast', 0.5);

    var spec = `/policy?act=${act}&cut=${cut}&lag=${lag}&tcase=${tcase}&kzone=${kzone}&vax=${vax}`;

    vegaEmbed(chart_box[0], spec, conf).then(function(result) {
        // access view as result.view
        chart_box.fadeTo('fast', 1.0);
    }).catch(console.error);
}

// render graphs from options
function updatePolicy() {
    var act = slide_act.val();
    var cut = slide_cut.val();
    var lag = slide_lag.val();
    var tcase = check_tcase.is(':checked');
    var kzone = check_kzone.is(':checked');
    var vax = check_vax.is(':checked');
    renderPolicy(act, cut, lag, tcase, kzone, vax);
}

//
// updating range indicators
//

var base_date = new Date(2020, 1, 15); // 2020-02-15
var day_msec = 1000*60*60*24;

function reflectNumber(inp) {
    var par = inp.closest('.option-item');
    var disp = par.find('.display-input');
    var val = inp.val();
    disp.text(val);
}

function reflectDate(inp) {
    var par = inp.closest('.option-item');
    var disp = par.find('.display-input');
    var val = inp.val();
    var start_date = new Date(base_date.getTime() + val*day_msec);
    var date_str = start_date.toDateString().slice(4);
    disp.text(date_str);
}

//
// key state tracking
//

var pressedKeys = {};
window.onkeyup = function(e) { pressedKeys[e.keyCode] = false; }
window.onkeydown = function(e) { pressedKeys[e.keyCode] = true; }

//
// policy updates
//

$('.option-slide').change(() => {
    if (!pressedKeys[37] && !pressedKeys[39]) {
        updatePolicy();
    }
});

$('.option-slide').keyup(() => {
    if (pressedKeys[37] || pressedKeys[39]) {
        updatePolicy();
    }
});

$('.option-check').change(updatePolicy);

//
// range indicators
//

$('.option-number').on('input', (event) => {
    var inp = $(event.target);
    reflectNumber(inp);
});

$('.option-date').on('input', (event) => {
    var inp = $(event.target);
    reflectDate(inp);
});

//
// profiles
//

prof_smash.click(() => {
    slide_act.val(0);
    slide_cut.val(0);
    check_kzone.prop('checked', true);
    check_vax.prop('checked', false);

    reflectNumber(slide_act);
    reflectNumber(slide_cut);

    updatePolicy();
});

prof_erad.click(() => {
    slide_act.val(45);
    slide_cut.val(10);
    check_kzone.prop('checked', true);
    check_vax.prop('checked', false);

    reflectNumber(slide_act);
    reflectNumber(slide_cut);

    updatePolicy();
});

prof_flat.click(() => {
    slide_act.val(50);
    slide_cut.val(50);
    check_kzone.prop('checked', false);
    check_vax.prop('checked', false);

    reflectNumber(slide_act);
    reflectNumber(slide_cut);

    updatePolicy();
});

prof_lfaire.click(() => {
    slide_act.val(100);
    slide_cut.val(0);
    check_kzone.prop('checked', false);
    check_vax.prop('checked', false);

    reflectNumber(slide_act);
    reflectNumber(slide_cut);

    updatePolicy();
});

prof_vax.click(() => {
    slide_act.val(60);
    slide_cut.val(10);
    check_kzone.prop('checked', false);
    check_vax.prop('checked', true);

    reflectNumber(slide_act);
    reflectNumber(slide_cut);

    updatePolicy();
});

//
// info popups
//

$('.info-button').mouseenter((e) => {
    var targ = $(e.target);
    if (targ.hasClass('info-icon')) {
        targ = targ.parent();
    }
    var pos = targ.position();
    var info = targ.attr('info');

    popup.html(info);
    var pop_width = popup.width();
    var pop_height = popup.height();

    var pop_left = pos.left - 0.5*pop_width + 3;
    var pop_top = pos.top - pop_height - 14;

    popup.css('left', pop_left);
    popup.css('top', pop_top);
    popup.fadeTo('fast', 0.9);
});

$('.info-button').mouseleave((e) => {
    popup.hide();
});

//
// initialize
//

// set current date
var curr_date = new Date();
var day_delta = Math.round((curr_date-base_date)/day_msec);
slide_lag.val(day_delta);

// sync up sliders
$('.option-number').each((i, el) => { reflectNumber($(el)); });
$('.option-date').each((i, el) => { reflectDate($(el)); });

// display policy
updatePolicy();
