# Steam Aspect Based Sentiment Analysis Web App
Sentiment analysis is the process of analyzing opinions towards a particular product, events, service, or topic to determine if the opinion is positive, negative, or neutral. There are three analysis level on sentiment analysis, document, sentence, and aspect. The aspect levels appear with the assumption that an opinion consist of sentiment and it's target (aspect). Therefore, in Aspect Based Sentiment Analysis (ABSA), there are two tasks that needs to be done, aspect extraction/classification and sentiment classification.

This  project is part of my undergraduate thesis where I created ABSA model using Convolutional Neural Networks (CNN) and compared it with the Long Short-Term Memory (LSTM). This repository contains the code for a Flask web that performs ABSA with model that already trained. The system will classify reviews into 9 different aspects:

1. Gameplay: The mechanics, playability, and control system of the game.
2. Graphics: The quality of the graphics, atmosphere, and the user interface of the game.
3. Story: The storyline, narrative, and character development of the game.
4. Sound: The quality of audio, songs, and sound effects in the game.
5. Developer: Reputation of the game developers on how well they listen to their players and providing updates.
6. Content: The amount of content and features available in the game.
7. Multiplayer: The quality of online features, server, and community in the gamee.
8. Performance: The performance and optimization of the game.
9. Value: The overall value and worth of the game and it's DLC.

## Results
The result shows that CNN works better than LSTM on aspect and sentiment classification.
<table><thead>
  <tr>
    <th> </th>
    <th colspan="4">Aspect Classification</th>
    <th colspan="4">Sentiment Classification</th>
  </tr></thead>
<tbody>
  <tr>
    <td> </td>
    <td colspan="2">CNN</td>
    <td colspan="2">LSTM</td>
    <td colspan="2">CNN</td>
    <td colspan="2">LSTM</td>
  </tr>
  <tr>
    <td>Accuracy</td>
    <td colspan="2">89%</td>
    <td colspan="2">88.5%</td>
    <td colspan="2">83.6%</td>
    <td colspan="2">80.7%</td>
  </tr>
  <tr>
    <td>Precision</td>
    <td colspan="2">87.7%</td>
    <td colspan="2">87.3%</td>
    <td colspan="2">82.3%</td>
    <td colspan="2">80.1%</td>
  </tr>
  <tr>
    <td>Recall</td>
    <td colspan="2">87.6%</td>
    <td colspan="2">87.3%</td>
    <td colspan="2">82.2%</td>
    <td colspan="2">79%</td>
  </tr>
  <tr>
    <td>F1-Score</td>
    <td colspan="2">87.7%</td>
    <td colspan="2">87.2%</td>
    <td colspan="2">82.1%</td>
    <td colspan="2">79.2%</td>
  </tr>
</tbody>
</table>

## Example
On the flask website, we only need to enter a game name. System will scrape reviews from Steam, preprocess, and classify it with trained model. Then it will show the sentiment of each aspect.

![](web_example/web_example.png)
![](web_example/result_1.png)
![](web_example/result_2.png)
![](web_example/result_3.png)
