# coinstac_ssr_fsl_2
Coinstac Code for Single-Shot Regression (ssr) on FreeSurfer Data

In COINSTAC, you can select any of the following Regions of Interest as Dependent variables in the regression. 

Left-Lateral-Ventricle
Left-Inf-Lat-Vent
Left-Cerebellum-White-Matter
Left-Cerebellum-Cortex
Left-Thalamus-Proper
Left-Caudate
Left-Putamen
Left-Pallidum
3rd-Ventricle
4th-Ventricle
Brain-Stem
Left-Hippocampus
Left-Amygdala
CSF
Left-Accumbens-area
Left-VentralDC
Left-vessel
Left-choroid-plexus
Right-Lateral-Ventricle
Right-Inf-Lat-Vent
Right-Cerebellum-White-Matter
Right-Cerebellum-Cortex
Right-Thalamus-Proper
Right-Caudate
Right-Putamen
Right-Pallidum
Right-Hippocampus
Right-Amygdala
Right-Accumbens-area
Right-VentralDC
Right-vessel
Right-choroid-plexus
5th-Ventricle
Optic-Chiasm
CC_Posterior
CC_Mid_Posterior
CC_Central
CC_Mid_Anterior
CC_Anterior
BrainSegVol
BrainSegVolNotVent
BrainSegVolNotVentSurf
lhCortexVol
rhCortexVol
CortexVol
lhCorticalWhiteMatterVol
rhCorticalWhiteMatterVol
CorticalWhiteMatterVol
SubCortGrayVol
TotalGrayVol
SupraTentorialVol
SupraTentorialVolNotVent
SupraTentorialVolNotVentVox
MaskVol
BrainSegVol-to-eTIV
MaskVol-to-eTIV
lhSurfaceHoles
rhSurfaceHoles
SurfaceHoles
EstimatedTotalIntraCranialVol

Tools: Python 3.6.5, coinstac-simulator 4.2.0

Steps
1) sudo npm i -g coinstac-simulator@4.2.0
2) git clone https://github.com/trendscenter/coinstac-ssr-fsl-2.git
3) cd coinstac_ssr_fsl_2
4) docker build -t ssr_fsl .
5) coinstac-simulator
