fpage = '''
<!DOCTYPE html>
<html>
<head>
<meta name="viewport" content="width=device-width, initial-scale=1">

<style>
  body {
    background-color: #E5E4E2;
  }

  h1 {
    color: maroon;
    margin-left: 40px;
  }
</style>
</head>

<body>

  <h1>Predict The Chance of Making the NBA playoffs</h1>
  <img src='https://img.bleacherreport.net/img/images/photos/003/804/554/hi-res-9a9716c6bb102c42af62340eeb4b0bea_crop_north.jpg?1556082334&w=3072&h=2048' height=200>


  <form id="row_info" action="data" method = "POST">

        <!-- I am not using the hidden field - just have it in case can think of some need for it later -->
        <input type='hidden' id='hidden1' value='hidden value'/>

        <p style="font-size:24pt;">Enter Wins <input style="font-size:18pt;" type = "text" name = "wins" placeholder="unknown"/></p>
        <p style="font-size:24pt;">Enter Loses <input style="font-size:18pt;" type = "text" name = "losses" placeholder="unknown"/></p>
        <p style="font-size:14pt;">In a Regular NBA season, Wins and Losses add up to 82</p>
        <p>
        <label for="finish" style="font-size:24pt;">Choose position in division:</label>
        <select id="finish" name="finish" style="font-size:18pt;">
          <option value="unknown">Unknown</option>
          <option value="1">1</option>
          <option value="2">2</option>
          <option value="3">3</option>
          <option value="4">4</option>
          <option value="5">5</option>
        </select>

        <p>
        <p style="font-size:24pt;">Enter Offensive Rating <input style="font-size:18pt;" type = "text" name = "ortg" placeholder="unknown"/></p>
        <p style="font-size:14pt;">An Estimate of points scored per 100 possesions. Typically between 79 and 118</p>
        <p style="font-size:24pt;">Enter Defensive Rating <input style="font-size:18pt;" type = "text" name = "drtg" placeholder="unknown"/></p>
        <p style="font-size:14pt;">An Estimate of points allowed per 100 possesions. Typically between 79 and 118</p>
        <p>

        <p>
        <p style="font-size:24pt;">Enter SRS (Strength of Schedule) <input style="font-size:18pt;" type = "text" name = "srs" placeholder="unknown"/></p>
        <p style="font-size:14pt;">Simple Rating System: a team rating that takes into account average point differential and strength of schedule. 
          The rating is denominated in points above/below average, where zero is average. Typically between -15 and 12</p>
        <p style="font-size:24pt;">Enter Pace <input style="font-size:18pt;" type = "text" name = "pace" placeholder="unknown"/></p>
        <p style="font-size:14pt;">An estimate of possessions per 48 minutes. Typically between 82 and 136</p>
        <p>

        <label for="league" style="font-size:24pt;">Choose league:</label>
        <select id="league" name="league" style="font-size:18pt;">
          <option value="unknown">Unknown</option>
          <option value="NBA">NBA</option>
          <option value="ABA">ABA</option>
          <option value="BAA">BAA</option>
        </select>

        <p><input style="font-size:24pt;" type = "submit" value = "Evaluate" /></p>
    </form>

    <script>
        <!-- toggle_image a bit of misnomer. It can be used to toggle tables on and off as well -->
        function toggle_image(im_id) {
            var state = document.getElementById(im_id).style.display;
            var new_state = 'inline';
            if (state=='inline'){new_state='none'};
            document.getElementById(im_id).style.display = new_state;
        }
    </script>

    <h1 onmouseover="toggle_image('image1')"
        onmouseout="toggle_image('image1')"
        style="display:inline" >  <!-- image1 is pipeline screenshot -->
      Results - probability of making the playoffs
    </h1>
    <ul>
      <!-- only showing threshold table for each method - might want to add tuning screenshot as well -->
      <li><h2 onmouseover="toggle_image('xgb')"
              onmouseout="toggle_image('xgb')"
              style="display:inline" >
            XGBoost alone: %xgb%    <!-- filled in once you have a prediction -->
          </h2></li>
      <p>
      <li><h2 onmouseover="toggle_image('knn')"
              onmouseout="toggle_image('knn')"
              style="display:inline" >
            KNN alone: %knn%
          </h2></li>
      <p>
      <li><h2 onmouseover="toggle_image('logreg')"
              onmouseout="toggle_image('logreg')"
              style="display:inline" >
            LogisticRegression alone: %logreg%
          </h2></li>
      <p>
      <li><h2 onmouseover="toggle_image('ann')"
              onmouseout="toggle_image('ann')"
              style="display:inline" >
            ANN alone: %ann%
          </h2></li>
      <p>
      <li><h2>Ensemble: %ensemble%</h2>
    </ul>

    <!-- Here are onmouseover images and tables - see above -->
    <div style="position:absolute; top:10px; left:500px">
      <!-- See notes above for hoops you have to jump through to get link to Drive image -->
      <img id="image1" style="display:none" src='https://drive.google.com/uc?export=view&id=1l16COqYsx25l5Lxb-JXiOH3JNdbyVcUX' height='400'>

      <!-- Below filled in with html table code when server starts up. They come from csv threshold files from your github. -->
      <div id="xgb" style='display:none'>
        %xgb_table%
        <hr>
        %xgb_lime_table%
      </div>
      <div id="knn" style='display:none'>
        %knn_table%
        <hr>
        %knn_lime_table%
      </div>
      <div id="logreg" style='display:none'>
        %logreg_table%
        <hr>
        %logreg_lime_table%
      </div>
      <div id="ann" style='display:none'>
        %ann_table%
        <hr>
        %ann_lime_table%
      </div>
    </div>



  </body>
  </html>
'''