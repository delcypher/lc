#!/bin/bash
#shell script to bring up VPN access to hercules & bluecrystal.
#This is intended to be run on a linux based system with pptpclient installed
#with the VPN already configured (use pptpsetup).

declare -x INTERFACE="ppp0"
declare -a ROUTES=(bluecrystalp2.bris.ac.uk hercules.star.bris.ac.uk)
declare -x VPN_NAME="bristol"

function usage()
{
	echo "
Usage: vpn.sh on - Activates VPN and sets up routes to hercules and bluecrystalphase2
       vpn.sh off - Disactivates VPN and routes.

	"
}

function wait-for-interface()
{
	DEVICE=$1;
	echo "Waiting for device...$DEVICE";
	while $( ifconfig $DEVICE 2> /dev/null 1>&2 ; if [ $? -eq 0 ]; then echo false; else echo true; fi); 
	do 
		sleep 1;
	done;

	echo "Interface $DEVICE visible, waiting for IP address..."
	COUNT=0
	while [ $COUNT -eq 0 ]; do
		COUNT=$( ifconfig ppp0 | grep -Ec 'inet addr:([0-9]{1,3}\.){3}([0-9]{1,3})' )
	done;
	
	echo "Interface $DEVICE has $( ifconfig ppp0 | grep -Eo 'inet addr:([0-9]{1,3}\.){3}([0-9]{1,3})' ) ";
}

if [ $# -ne 1 ]; then
	usage;
	exit 1;
fi

if [ $1 = "on" ]; then
	echo "Switching VPN on"
	pon $VPN_NAME
	wait-for-interface $INTERFACE
	for host in ${ROUTES[*]}; do
		echo "Adding route...$host"
		route add -host $host dev $INTERFACE
	done
	exit 0;
fi

if [ $1 = "off" ]; then
	echo "Switching VPN off";
	wait-for-interface $INTERFACE
	for host in ${ROUTES[*]}; do
		echo "Deleting route...$host"
		route del -host $host dev $INTERFACE
	done;
	poff $VPN_NAME
	exit 0;

fi
