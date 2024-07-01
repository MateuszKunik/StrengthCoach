StrengthApp
==============================

An AI-powered web application that can serve as your strength training coach, guiding you through exercises like squats, bench presses or deadlifts.


-----------
```latex

\documentclass[12pt,a4paper]{article}
\usepackage[cp1250]{inputenc}	
\usepackage{graphicx}

\usepackage{geometry}
\geometry{margin=2.5cm}

\usepackage{multicol}
\usepackage{multirow}
\pagestyle{plain}
\usepackage{cite}

\usepackage{amsmath, amsfonts, amssymb}
\usepackage{enumitem}
\usepackage{caption}

\usepackage{float}
\usepackage{tabularx}

\usepackage{titlesec}
\titleformat{\section}{\bfseries}{}{0pt}{}
\titlespacing\section{0pt}{4pt}{0pt}

\titleformat{\subsection}{\bfseries}{}{0pt}{}
\titlespacing*{\subsection}{0pt}{4pt}{0pt}

%\setlength{\parindent}{0pt}

%\usepackage{enumerate}
%\usepackage{amsthm}
%\usepackage{latexsym}
%\usepackage{pgfplots}
%\usepackage{tikz}
%\usepackage{mathtools}
%\usepackage{hyperref}
%\usepackage{natbib}
%\usepackage{subcaption}



  
\title{The use of deep neural networks in strength sports on the example of One-Repetition Maximum prediction}
\author{Mateusz Michał Kunik}
\date{\today}

\begin{document}

\maketitle

\begin{abstract}
The article presents an innovative approach to predicting the One-Repetition Maximum (in short 1-RM) parameter, commonly used in strength sports. The proposed method combines deep neural networks with computer vision techniques. The study focuses on a comprehensive analysis of strength exercise technique based on audiovisual materials. Traditional methods of measuring 1-RM and their limitations are discussed, alongside a newly proposed method that integrates advanced image analysis with athletes' training specifics.

\par In the study, computer vision algorithms were applied to estimate body pose, followed by the use of advanced neural network architectures to analyze the obtained data. A dedicated dataset containing recordings of study participants was developed. The goal of this method is to increase the accuracy of 1-RM prediction while eliminating the inconveniences of traditional methods. The article discusses potential practical applications and future research directions in applying artificial intelligence to strength sports.
\end{abstract}

\begin{multicols}{2}
\section{Introduction to One-Repetition Maximum}
\noindent One-Repetition Maximum (1-RM) is a parameter primarily used in strength sports to determine the maximum weight an individual can lift for a single repetition of a chosen exercise \cite{marchese2005essential, haff2015essentials}. It is a key indicator used to assess muscular strength and monitor training progress. Additionally, accurate measurement allows for adjusting training intensity according to individual capabilities and training goals. This approach ensures optimal muscle stimulation while minimizing the risk of injury \cite{Suchomel2021TrainingFM, Hunter1995RelativeTI, hickmott2022effect}.

\par 1-RM is predominantly utilized in sports such as powerlifting, Olympic weightlifting, and strongman competitions. Exercises commonly assessed using this indicator include back squat, bench press, and deadlift. Moreover, it is also used in the athletic preparation of sports such as track and field, team sports, and combat sports. In these disciplines, high muscular strength can contribute to increased speed, explosiveness, and overall physical performance of athletes \cite{Jeong2023TheSO, Judge2011DesigningAE}.


\section{Measurement, Estimation, and Prediction of the 1-RM}
\noindent The One-Repetition Maximum can be assessed in two ways: directly - by performing a maximal load test, and indirectly - through estimation or prediction using sub-maximal loads.


\par In the strength sports community, athletes and enthusiasts most commonly determine 1-RM directly by performing a maximal load test. Due to the nature of conducting this measurement, this method is also referred to as trial and error.

\par The measurement process proceeds as follows. The athlete warms up, gradually increasing the weight with each subsequent attempt. They rest for several, even up to several dozen minutes between each set. As the weight increases, the number of repetitions per set decreases to one. The athlete concludes the test when unable to perform a repetition at all or with proper technique. The 1-RM is equal to the weight lifted in the last successful attempt. Throughout the test, external individuals such as coaches or judges assess the technique and validate each attempt.

\begin{itemize}[leftmargin=*, itemsep=0pt, topsep=5pt, label={}]
    \item Advantages:
    \begin{itemize}[leftmargin=*, topsep=0pt]
        \item The most precise method for assessing maximal load.
        \item Relatively easy to apply.
    \end{itemize}
    \item Disadvantages:
    \begin{itemize}[leftmargin=*, topsep=0pt]
        \item Involves risk of injury, especially with improper technique.
        \item Places high strain on the nervous system and negatively impacts physical health.
        \item Time-consuming and requires proper preparation.
        \item Can induce feelings of overwhelm and concern due to relatively heavy loads.
    \end{itemize}
\end{itemize}


\par An alternative to the maximal load test is the use of indirect methods, among which two different approaches are distinguished. The advantage of estimating and predicting the 1-RM parameter over the direct method is primarily the reduction of the risk of injury and damage due to the use of sub-maximal loads. Additionally, they limit the negative impact on the athlete's physical health and nervous system. They also shorten the time needed to conduct measurements. Furthermore, using loads similar to those used in regular training sessions positively affects the athlete's psyche by increasing their confidence. It should be noted that the choice of indirect methods does not eliminate the disadvantages of direct measurement, but merely limits them.

\par A more commonly used approach, mainly due to its simplicity but also due to a longer period of research and accumulation of knowledge, is the estimation method. Specifically, these are mathematical models of low complexity that describe the relationship between weight and the number of repetitions and maximum muscle strength.

\par Initially, the measurement process may resemble a maximal load test - the athlete warms up, gradually increasing the weight while limiting the number of repetitions. This time, the athlete stops at a sub-maximal load, which means sufficiently heavy but below the maximal load. With an appropriately chosen weight, the athlete performs the maximum number of repetitions while maintaining proper exercise technique. Based on the results obtained, the 1-RM parameter is estimated using developed mathematical models.

\renewcommand{\arraystretch}{1.194}
\begin{table}[H]
    \centering
    \begin{tabularx}{\columnwidth}{p{2.25cm} | X}
        \hline
        Autor & Formuła \\
        \hline
        Brzycki \cite{brzycki1993strength} & $w\frac{36}{37 - r}$\\
        Epley \cite{epley1985poundage} & $w(1+\frac{r}{30})$\\
		Lander \cite{landers1984maximum} & $w(1.013 - 0.0267123r)^{-1}$\\        
        Lombardi \\ \hspace{1pt} \cite{lombardi1989beginning} & $wr^{0.1}$\\
        Mayhew \\ \hspace{1pt} et al. \cite{mayhew1992relative} & $w(0.522 + 0.419e^{-0.055r})^{-1}$\\
        O'Conner \\ \hspace{1pt} et al. \cite{o1989weight} & $w(1+0.025r)$\\
        \hline
    \end{tabularx}
    \caption{It presents selected models estimating the 1-RM parameter. The variable w denotes the weight in any unit of mass, and the variable r corresponds to the number of repetitions.}
    \label{tabela1}
\end{table}



% NEW PAGE !!!!!
%\newpage 


\par Currently, the most commonly used formula is the one proposed by B. Epley in 1985 \cite{epley1985poundage} or a few years later by M. Brzycki \cite{brzycki1993strength}. It is worth mentioning that the Epley and Brzycki formulas yield identical results for 10 repetitions. However, for fewer than 10 repetitions, the Epley formula returns slightly higher estimated values. Based on numerous scientific studies, over time, other mathematical models estimating the 1-RM value have also been developed. Some of these are included in Table \ref{tabela1}. Improved estimators have become the starting point for further research and meta-analyses of 1-RM assessment \cite{mayhew2008accuracy, knutzen1999validity, chapman1998225, reynolds2006prediction, lesuer1997accuracy, wood2002accuracy}.

\begin{itemize}[leftmargin=*, itemsep=0pt, topsep=5pt, label={}]
    \item Disadvantages:
    \begin{itemize}[leftmargin=*, topsep=0pt]
        \item The estimation method is less accurate than the maximal load test.
        \item Most models are based on studies conducted on experienced athletes. Therefore, experienced individuals who know their capabilities better benefit more.
        \item Estimation results strongly depend on the chosen mathematical model.
    \end{itemize}
\end{itemize}

\par A new and significantly more technologically advanced approach is the method of predicting 1-RM using the relationship between load and movement velocity (Load-Velocity Relationship, or LVR) \cite{picerno20161rm, thompson2021novel}. This method involves the use of advanced sensors, such as accelerometers, to precisely measure the concentric phase velocity of the movement during exercise. These data are then analyzed to create a curve interpolating the relationship between load and movement velocity. Assuming that movement velocity approaches zero as the load increases, velocities for heavier loads are extrapolated, including the maximum weight the athlete can lift for one repetition \cite{gonzalez2010movement, sanchez2011velocity, Jidovtseff2011UsingTL, marston2022load, weakley2021velocity}. An example of such interpolation with extrapolation is shown in Fig. \ref{figura1}.


\begin{figure}[H]
    \centering
    \includegraphics[width=0.5\textwidth]{plot.jpg}
    \caption{Visualization of the 1-RM prediction method using linear and quadratic models. Interpolation predicts the load up to $80\%$ of the estimated 1-RM based on measurements, and extrapolation estimates the maximum load based on previous observations. The dotted lines represent the linear model, while the dashed lines represent the quadratic model \cite{thompson2021novel}.}
    \label{figura1}
\end{figure}

\par The measurement process differs slightly from the previously presented methods. This time, the athlete performs a short warm-up and then proceeds to measure movement velocity in several sets with progressively increasing loads. Typically, loads ranging from $50\%$ to even $90\%$ of the estimated 1-RM are used. It is important to remember that the more measurements the athlete performs, the more precise the 1-RM prediction should be.

\begin{itemize}[leftmargin=*, itemsep=0pt, topsep=5pt, label={}]
    \item Disadvantages:
    \begin{itemize}[leftmargin=*, topsep=0pt]
    	\item The predicted values significantly overestimate the actual 1-RM.
        \item The need for specialized equipment to measure velocity and requires the skills to operate the equipment and software.
        \item The forecasting results depend on the measuring device used.
    \end{itemize}
\end{itemize}

\par In summary, there are two main ways to assess the 1-RM indicator. Direct measurement, by conducting a maximal load test, is the most precise method, allowing for an accurate determination of the current 1-RM. Unfortunately, it has several significant drawbacks, which are well addressed by indirect methods. Methods such as estimation based on weight and number of repetitions and prediction using the simple relationship between load and movement velocity are good solutions. However, as previously mentioned, they have numerous disadvantages, which we believe stem from the low complexity of the models and the lack of application of modern solutions provided by deep neural networks.


\section{Performance}
\noindent The starting point for research aimed at improving the quality of assessing the 1-RM parameter is performance-based prediction - that is, predicting based on the execution of a full set or a single repetition of a chosen exercise.

\par Performance is defined as a set of characteristics that influence the overall execution of an approach and provide information about the athlete's maximum muscle strength. We distinguish two types of performance characteristics: observable and unobservable. Observable characteristics are elements of performance that can be directly observed in the athlete during exercise execution. These include speed, body movement trajectory, and stability of the person performing the exercise. These characteristics are all observable through audiovisual material. On the other hand, unobservable characteristics include factors such as training level, training experience, and the type of training program specific to the athlete. In other words, these are characteristics that indirectly influence performance. We can consider a personal questionnaire as a carrier of such information.

\par Predicting the 1-RM parameter based on performance should be divided into several independent stages. Below, we present a proposal for the successive steps, with the complete diagram depicted in Figure \ref{figura2}.

\begin{enumerate}[leftmargin=*, itemsep=0pt, topsep=5pt]
    \item \textbf{Collecting audiovisual material:} The first step involves recording the athlete performing a series of repetitions of the chosen exercise. The recording should adhere to specific guidelines, which will be discussed later.
    \item \textbf{Estimating body position:} The audiovisual recording is then processed by a body position estimation module. Advanced models such as MediaPipe, OpenPose, DeepLabCut, or PoseNet can be used for this purpose. These models detect points on the athlete's body during exercise execution.
    \item \textbf{Gathering personal data:} Additional information about the athlete should be collected through interviews or personal questionnaires. This data should include details like training level, experience, and training program type. It's important to ensure data anonymization.
    \item \textbf{Initial data processing:} The collected data is merged and transformed to obtain a numerical representation in the form of a tensor. A tensor allows for efficient storage and processing of multidimensional data.
    \item \textbf{Predicting $\%$ 1-RM:} After data processing, the tensor is input into a prediction module based on deep neural networks. These networks analyze the data received, from which the model determines what percentage of their one-repetition maximum the athlete lifted in the provided recording.
    \item \textbf{Calculating 1-RM:} Based on the prediction from the model and information about the weight used during the exercise, the athlete's estimated 1-RM is calculated.
\end{enumerate}

\begin{figure*}[ht]
    \centering
    \includegraphics[width=\textwidth]{fullprocess_2.jpg}
    \caption{Visualization of the performance-based 1-RM prediction process.}
    \label{figura2}
\end{figure*}

\par From the athlete's perspective, the entire prediction process involves just three steps. The athlete should record several sets of repetitions at sub-maximal load. A standard smartphone camera can be used for recording the exercise. Next, they should upload the prepared audiovisual material to a dedicated mobile application, which also includes a personal questionnaire. The final step is completing the mentioned personal questionnaire.

\par Performance analysis can be effective in predicting the 1-RM parameter by utilizing both observable and unobservable characteristics. As we have discussed, this process relies on advanced image analysis techniques such as body position estimation, integrated with personal data.

\section{Dataset}
\noindent To effectively analyze performance and thereby predict the 1-RM parameter, a deep neural network model trained on an appropriate dataset is necessary. Due to the specificity of the method we have adopted and the innovative approach to the issue, it was impossible to find a suitable dataset. Therefore, we were obligated to develop and prepare our own dataset.

\par The data was collected from two main sources: personal questionnaires and audiovisual recordings. The questionnaires contain demographic information and training details of the participants. Meanwhile, the audiovisual recordings are from sessions where participants performed back squat exercises in a maximal load test format.

\par The study involved 15 volunteers, including 12 men and 3 women. The participants' skill levels varied: 2 were beginners, 9 were intermediate, and 4 were advanced. The data includes various information such as age, gender, height, weight, skill level, training experience, equipment availability, type of training program, weekly training frequency, and participation in powerlifting competitions.

\par The data were collected by having participants fill out surveys and recording them while performing the exercise. The procedure for collecting audiovisual materials resembled that of a maximal load test. Participants were tasked with performing several series with different loads. Starting from lighter series, through moderate-intensity ones, up to heavy single repetitions with sub-maximal and maximal loads. Participants performed successive attempts until failure. The test ended only when the participant could not perform a single repetition. After a short break, participants performed the maximum number of repetitions (As Many Repetitions As Possible, AMRAP) with a load approximately equal to $75\%$ of their current 1-RM. Recordings were made with three cameras, capturing exercises from different angles - front, left, and right sides at a 45-degree angle. Cameras were securely mounted on tripods at hip height of the person being recorded. For safety reasons, participants were provided with spotting. Additionally, all repetitions were meticulously analyzed for technical correctness by a powerlifting coach.

\par The initial data preprocessing involved segmenting the audiovisual material into short recordings containing only one repetition of the back squat. As a result, we obtained over 1300 samples of recordings. To preserve key information and maintain consistency with the raw research data, we implemented a labeling system for the recordings. Each recording was assigned a unique identifier that sequentially includes the participant ID, series number, repetition number within the series, load, indication of whether the attempt was successful, and camera position. An example label would be: $006\_06\_03\_01\_080\_1\_C$.

\par During data collection and processing, several issues related to the quality of audiovisual recordings, improper camera settings, or missed attempts were encountered. To ensure compliance with data protection regulations, all data were collected with participants' consent, ensuring anonymity and proper storage.

\section{Estymacja Pozy Ciała}

\section{Sieci Neuronowe}

\section{Rekurencyjne sieci neuronowe}

\section{Grafowe sieci splotowe}

\section{Analiza wyników}

\section{Wyzwania i przyszłe kierunki badań}

\section{Podsumowanie i wnioski}


\end{multicols}

\newpage
\bibliographystyle{unsrt}
\bibliography{D:\\Matematyka\\mybibliography}

\end{document}




```bibtex
@book{marchese2005essential,
  title={The Essential Guide to Fitness for the Fitness Instructor},
  author={Marchese, Rosemary and Hill, Andrew},
  year={2005},
  publisher={Pearson Education},
  address={Frenchs Forest, N.S.W.}
}

@article{Suchomel2021TrainingFM,
  title={Training for Muscular Strength: Methods for Monitoring and Adjusting Training Intensity},
  author={Timothy J. Suchomel and Sophia Nimphius and Christopher R. Bellon and William Guy Hornsby and Michael H. Stone},
  journal={Sports Medicine},
  year={2021},
  volume={51},
  pages={2051 - 2066},
  url={https://api.semanticscholar.org/CorpusID:235364256}
}

@article{Hunter1995RelativeTI,
  title={Relative Training Intensity and Increases in Strength in Older Women},
  author={Gary R. Hunter and Margarita S. Treuth},
  journal={Journal of Strength and Conditioning Research},
  year={1995},
  volume={9},
  pages={188–191},
  url={https://api.semanticscholar.org/CorpusID:143810641}
}

@article{Jeong2023TheSO,
  title={The Squat One Repetition Maximum May Not Be the Best Indicator for Speed-Related Sports Performance Improvement in Elite Male Rugby Athletes},
  author={Yeunchang Jeong and Hyung-pil Jun and Yu-Lun Huang and Eunwook Chang},
  journal={Applied Sciences},
  year={2023},
  url={https://api.semanticscholar.org/CorpusID:266461241}
}

@article{Judge2011DesigningAE,
  title={Designing an Effective Preactivity Warm-up Routine for the 1 Repetition Maximum Back Squat},
  author={Lawrence W. Judge and Joshua N Wildeman and David Bellar},
  journal={Strength and Conditioning Journal},
  year={2011},
  volume={33},
  pages={88-90},
  url={https://api.semanticscholar.org/CorpusID:70488651}
}


@article{mayhew2008accuracy,
  title={Accuracy of prediction equations for determining one repetition maximum bench press in women before and after resistance training},
  author={Mayhew, Jerry L and Johnson, Blair D and LaMonte, Michael J and Lauber, Dirk and Kemmler, Wolfgang},
  journal={The Journal of Strength \& Conditioning Research},
  volume={22},
  number={5},
  pages={1570--1577},
  year={2008},
  publisher={LWW}
}

@article{lesuer1997accuracy,
  title={The accuracy of prediction equations for estimating 1-RM performance in the bench press, squat, and deadlift},
  author={LeSuer, Dale A and McCormick, James H and Mayhew, Jerry L and Wasserstein, Ronald L and Arnold, Michael D and others},
  journal={Journal of strength and conditioning research},
  volume={11},
  pages={211--213},
  year={1997},
  publisher={HUMAN KINETICS PUBLISHERS, INC.}
}

@article{knutzen1999validity,
  title={Validity of 1RM prediction equations for older adults},
  author={Knutzen, Kathleen M and BRILLA, LORRAINE R and Caine, Dennis},
  journal={The Journal of Strength \& Conditioning Research},
  volume={13},
  number={3},
  pages={242--246},
  year={1999},
  publisher={LWW}
}

@article{chapman1998225,
  title={The 225--1b reps-to-fatigue test as a submaximal estimate of 1-RM bench press performance in college football players},
  author={Chapman, Paul P and Whitehead, James R and Binkert, Ronald H},
  journal={The Journal of Strength \& Conditioning Research},
  volume={12},
  number={4},
  pages={258--261},
  year={1998},
  publisher={LWW}
}

@article{reynolds2006prediction,
  title={Prediction of one repetition maximum strength from multiple repetition maximum testing and anthropometry},
  author={Reynolds, Jeff M and Gordon, Toryanno J and Robergs, Robert A},
  journal={The Journal of Strength \& Conditioning Research},
  volume={20},
  number={3},
  pages={584--592},
  year={2006},
  publisher={LWW}
}

@book{haff2015essentials,
  title={Essentials of strength training and conditioning 4th edition},
  author={Haff, G Gregory and Triplett, N Travis},
  year={2015},
  publisher={Human kinetics}
}

@article{epley1985poundage,
  title={Poundage chart},
  author={Epley, B},
  journal={Boyd Epley Workout. Lincoln, NE: Body Enterprises},
  volume={86},
  year={1985}
}

@article{brzycki1993strength,
  title={Strength testing—predicting a one-rep max from reps-to-fatigue},
  author={Brzycki, Matt},
  journal={Journal of physical education, recreation \& dance},
  volume={64},
  number={1},
  pages={88--90},
  year={1993},
  publisher={Taylor \& Francis}
}

@article{landers1984maximum,
  title={Maximum based on reps},
  author={Landers, J},
  journal={National Strength \& Conditioning Association Journal},
  volume={6},
  number={6},
  pages={60},
  year={1984},
  publisher={Ovid Technologies (Wolters Kluwer Health)}
}

@article{lombardi1989beginning,
  title={Beginning weight training: the safe and effective way},
  author={Lombardi, V Patteson},
  journal={(No Title)},
  year={1989}
}

@article{mayhew1992relative,
  title={Relative muscular endurance performance as a predictor of bench press strength in college men and women},
  author={Mayhew, Jerry L and Ball, Thomas E and Arnold, Micheal D and Bowen, Jack C},
  journal={The Journal of Strength \& Conditioning Research},
  volume={6},
  number={4},
  pages={200--206},
  year={1992},
  publisher={LWW}
}

@book{o1989weight,
  title={Weight Training Today},
  author={O'Connor, R. and Simmons, J. and O'Shea, P.},
  isbn={9780314689511},
  lccn={89009115},
  url={https://books.google.pl/books?id=eHqfwq6o18oC},
  year={1989},
  publisher={West}
}

@article{richens2014relationship,
  title={The relationship between the number of repetitions performed at given intensities is different in endurance and strength trained athletes},
  author={Richens, Ben and Cleather, Daniel J},
  journal={Biology of sport},
  volume={31},
  number={2},
  pages={157--161},
  year={2014},
  publisher={Termedia}
}

@article{hickmott2022effect,
  title={The effect of load and volume autoregulation on muscular strength and hypertrophy: A systematic review and meta-analysis},
  author={Hickmott, Landyn M and Chilibeck, Philip D and Shaw, Keely A and Butcher, Scotty J},
  journal={Sports medicine-open},
  volume={8},
  number={1},
  pages={9},
  year={2022},
  publisher={Springer}
}

@article{gonzalez2010movement,
  title={Movement velocity as a measure of loading intensity in resistance training},
  author={Gonz{\'a}lez-Badillo, Juan J and S{\'a}nchez-Medina, L},
  journal={International journal of sports medicine},
  pages={347--352},
  year={2010},
  publisher={{\copyright} Georg Thieme Verlag KG Stuttgart{\textperiodcentered} New York}
}

@article{sanchez2011velocity,
  title={Velocity loss as an indicator of neuromuscular fatigue during resistance training.},
  author={Sanchez-Medina, Luis and Gonz{\'a}lez-Badillo, Juan Jos{\'e}},
  journal={Medicine and science in sports and exercise},
  volume={43},
  number={9},
  pages={1725--1734},
  year={2011}
}

@article{Jidovtseff2011UsingTL,
  title={Using the load-velocity relationship for 1RM prediction},
  author={Boris Jidovtseff and Nigel K Harris and J. M. Crielaard and John B. Cronin},
  journal={Journal of Strength and Conditioning Research},
  year={2011},
  volume={25},
  pages={267-270},
  url={https://api.semanticscholar.org/CorpusID:207502224}
}

@article{marston2022load,
  title={Load-velocity relationships and predicted maximal strength: A systematic review of the validity and reliability of current methods},
  author={Marston, Kieran J and Forrest, Mitchell RL and Teo, Shaun YM and Mansfield, Sean K and Peiffer, Jeremiah J and Scott, Brendan R},
  journal={PLoS One},
  volume={17},
  number={10},
  pages={e0267937},
  year={2022},
  publisher={Public Library of Science San Francisco, CA USA}
}

@article{thompson2021novel,
  title={A novel approach to 1RM prediction using the load-velocity profile: a comparison of models},
  author={Thompson, Steve W and Rogerson, David and Ruddock, Alan and Greig, Leon and Dorrell, Harry F and Barnes, Andrew},
  journal={Sports},
  volume={9},
  number={7},
  pages={88},
  year={2021},
  publisher={MDPI}
}

@article{weakley2021velocity,
  title={Velocity-based training: From theory to application},
  author={Weakley, Jonathon and Mann, Bryan and Banyard, Harry and McLaren, Shaun and Scott, Tannath and Garcia-Ramos, Amador},
  journal={Strength \& Conditioning Journal},
  volume={43},
  number={2},
  pages={31--49},
  year={2021},
  publisher={LWW}
}

@article{wood2002accuracy,
  title={Accuracy of seven equations for predicting 1-RM performance of apparently healthy, sedentary older adults},
  author={Wood, Terry M and Maddalozzo, Gianni F and Harter, Rod A},
  journal={Measurement in physical education and exercise science},
  volume={6},
  number={2},
  pages={67--94},
  year={2002},
  publisher={Taylor \& Francis}
}

@article{picerno20161rm,
  title={1RM prediction: a novel methodology based on the force--velocity and load--velocity relationships},
  author={Picerno, Pietro and Iannetta, Danilo and Comotto, Stefania and Donati, Marco and Pecoraro, Fabrizio and Zok, Mounir and Tollis, Giorgio and Figura, Marco and Varalda, Carlo and Di Muzio, Davide and others},
  journal={European Journal of Applied Physiology},
  volume={116},
  pages={2035--2043},
  year={2016},
  publisher={Springer}
}

@article{johansson1973visual,
  title={Visual perception of biological motion and a model for its analysis},
  author={Johansson, Gunnar},
  journal={Perception \& psychophysics},
  volume={14},
  pages={201--211},
  year={1973},
  publisher={Springer}
}

@article{shi2020skeleton,
  title={Skeleton-based action recognition with multi-stream adaptive graph convolutional networks},
  author={Shi, Lei and Zhang, Yifan and Cheng, Jian and Lu, Hanqing},
  journal={IEEE Transactions on Image Processing},
  volume={29},
  pages={9532--9545},
  year={2020},
  publisher={IEEE}
}

@book{hamilton2020graph,
  title={Graph representation learning},
  author={Hamilton, William L},
  year={2020},
  publisher={Morgan \& Claypool Publishers}
}

@inproceedings{yan2018spatial,
  title={Spatial temporal graph convolutional networks for skeleton-based action recognition},
  author={Yan, Sijie and Xiong, Yuanjun and Lin, Dahua},
  booktitle={Proceedings of the AAAI conference on artificial intelligence},
  volume={32},
  number={1},
  year={2018}
}

@article{kipf2016semi,
  title={Semi-supervised classification with graph convolutional networks},
  author={Kipf, Thomas N and Welling, Max},
  journal={arXiv preprint arXiv:1609.02907},
  year={2016}
}

@inproceedings{soo2017interpretable,
  title={Interpretable 3d human action analysis with temporal convolutional networks},
  author={Soo Kim, Tae and Reiter, Austin},
  booktitle={Proceedings of the IEEE conference on computer vision and pattern recognition workshops},
  pages={20--28},
  year={2017}
}

@article{bahdanau2014neural,
  title={Neural machine translation by jointly learning to align and translate},
  author={Bahdanau, Dzmitry and Cho, Kyunghyun and Bengio, Yoshua},
  journal={arXiv preprint arXiv:1409.0473},
  year={2014}
}

@article{velickovic2017graph,
  title={Graph attention networks},
  author={Velickovic, Petar and Cucurull, Guillem and Casanova, Arantxa and Romero, Adriana and Lio, Pietro and Bengio, Yoshua and others},
  journal={stat},
  volume={1050},
  number={20},
  pages={10--48550},
  year={2017}
}

@inproceedings{li2019actional,
  title={Actional-structural graph convolutional networks for skeleton-based action recognition},
  author={Li, Maosen and Chen, Siheng and Chen, Xu and Zhang, Ya and Wang, Yanfeng and Tian, Qi},
  booktitle={Proceedings of the IEEE/CVF conference on computer vision and pattern recognition},
  pages={3595--3603},
  year={2019}
}

@article{wang2019comparative,
  title={A comparative review of recent kinect-based action recognition algorithms},
  author={Wang, Lei and Huynh, Du Q and Koniusz, Piotr},
  journal={IEEE Transactions on Image Processing},
  volume={29},
  pages={15--28},
  year={2019},
  publisher={IEEE}
}

@article{le2022skeleton,
  title={Skeleton-based human action recognition using spatio-temporal attention graph convolutional networks},
  author={Le, Manh Cuong},
  year={2022}
}

@inproceedings{Niewiadomski2008DeterminationAP,
  title={Determination and Prediction of One Repetition Maximum (1RM): Safety Considerations},
  author={Wiktor Niewiadomski and Dorota Laskowska and Anna Gąsiorowska and Gerard Cybulski and Anna Strasz and J{\'o}zef Langfort},
  year={2008},
  url={https://api.semanticscholar.org/CorpusID:70507959}
}




```markdown
Project Organization
------------

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── docs               <- A default Sphinx project; see sphinx-doc.org for details
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to download or generate data
    │   │   └── make_dataset.py
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │   └── build_features.py
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├── predict_model.py
    │   │   └── train_model.py
    │   │
    │   └── visualization  <- Scripts to create exploratory and results oriented visualizations
    │       └── visualize.py
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.readthedocs.io


--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
