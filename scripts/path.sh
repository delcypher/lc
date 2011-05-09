#Script that adds the executables to the work path
# run as ``source path.sh'' or ``. path.sh'' from the bash shell

declare -i INPATH=0

addpath=$( cd ${BASH_SOURCE[0]%/*} ; echo "$PWD" )

#strip off '/scripts'
addpath=${addpath%/*}


#check if path already present
tempIFS=$IFS
IFS=$':'
#Extract each path seperated by a ':' character and compare
for singlepath in $PATH ; do
	if [ "$singlepath" = "$addpath" ]; then
		INPATH=1;
	fi
done

if [ $INPATH -eq 0 ]; then
	export PATH=$PATH:$addpath
	echo "Added $addpath to PATH environmental variable"
else
	echo "$addpath is already in PATH environmental variable"
fi

#Set IFS back...
IFS=$tempIFS

#add alias for view-state.sh
alias view-state="$addpath/scripts/view-state.sh" && echo "Added view-state alias"

#add alias for view-energy-density-colour-map.sh
alias view-edcm="$addpath/scripts/view-energy-density-colour-map.sh" && echo "Added view-edcm alias"

#add alias for view-energy-density-3d.sh
alias view-ed3d="$addpath/scripts/view-energy-density-3d.sh" && echo "Added view-ed3d alias"
