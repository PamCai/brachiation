import numpy as np
#import matplotlib as mpl
import matplotlib.pyplot as plt

def rheo_calc(c,x,M,ku,kb,N,P,omega,
			  plot_calc=False,plot_calc_all=False):

	""" Analytically calculate the rheological prediction 
	of Brachiation Theory.

	Parameters
	----------
	c : float
	    Concentration of polymer 
	x : float
	    Drag coefficient
	M : int
	    Number of stickers along chain
	ku : float
	     Unbinding rate constant
	kb : float
		 Binding rate constant
	N : int
		Number of total monomers on chain
	P : int
	    Number of modes to compute (should be at least 300)
	omega : array of floats
	        Frequency range
	plot_calc : boolean
			    Plots each iteration in calculation
	plot_calc_all : boolean
	                Plots final rheological prediction

	Returns
	-------
	G : array of floats
		Rheological predictions of Brachiation Theory 
	G0 : array of floats
		 Non-dimensionalized starting point for the high p limit
	"""
	
	# calculate the binding probability
	pb = kb*c*M/(ku+kb*c*M)

	# get number of frequencies and highest frequency
	w_n = len(omega)
	w_hi = np.max(omega)
	# set max n to start from and normal mode vectors
	n_max = np.maximum(np.ceil(0.00000001*w_hi/ku),50.)
	p_vec = np.linspace(1.,P,P)
	kp = np.diag(np.float_power(p_vec,2.)) 

	# set up initial Cpp matrix at high n
	E1 = np.outer(np.float_power(np.float_power(p_vec,2.)+n_max*ku,-1.),np.ones((1,w_n),dtype=np.complex))
	F1 = np.outer(np.float_power(n_max*ku+np.float_power(p_vec,2.),-2.),(1j*omega))
	Cpp = E1 - F1
	K0 = c*np.sum(Cpp,axis=0)
	G0 = np.multiply((1j*omega+n_max*ku),K0/c)

	# set up Phi_pp'
	def phip(ind,p,M):
		phip = np.zeros(M)
		for i in ind:
			phip[i-1] = np.cos(p*i/M)
		return phip

	Phi_pp = np.zeros((P,P))
	ind = np.array(range(M))+1
	for l in range(P):
		for m in range(P):
			Phi_pp[l,m] = np.sum(np.multiply(phip(ind,(l+1),M),phip(ind,(m+1),M)))/M

	# start at n_max and step back to 0
	K = np.zeros(w_n,dtype=np.complex)
	G = np.zeros_like(K)
	for j in range(np.int(n_max+1)):
		n = n_max-j
		for w in range(w_n):
			den = (1j*omega[w]+n*ku)*x*np.identity(P) + (1j*omega[w]+n*ku)*pb*Phi_pp*M*K[w] + kp
			num = x*np.identity(P) + pb*Phi_pp*M*K[w]
			C_pp = np.linalg.solve(den,num)
			K[w] = c*np.sum(np.diag(C_pp))
			G[w] = (1j*omega[w]+n*ku)*K[w]/c
		print(int(n))
		if plot_calc:
			plt.plot(omega,np.real(G),'r')
			plt.plot(omega,np.imag(G),'b')
			plt.plot(omega,np.real(G0),'r:')
			plt.plot(omega,np.imag(G0),'b:')
			plt.ylabel(r'$NG/(ck_{B}T)$')
			plt.xlabel(r'$\tau_{R} \omega$')
			plt.yscale('log')
			plt.xscale('log')
			plt.show(block=False)
			plt.pause(0.1)
			plt.close()

	# calculate Rouse model limit
	if plot_calc_all:
		GR = np.zeros(w_n)
		for p in range(P):
			GR = GR + 1j*omega/(np.float_power((p+1),2.)+1j*omega)
		plt.plot(omega,np.real(G),'r')
		plt.plot(omega,np.imag(G),'b')
		plt.plot(omega,np.real(GR),'r--',markersize=1)
		plt.plot(omega,np.imag(GR),'b--',markersize=1)
		plt.plot(np.array([ku,ku]),np.array([1.e-6,1.e2]),'k:')
		plt.ylabel(r'$NG/(ck_{B}T)$')
		plt.xlabel(r'$\tau_{R} \omega$')
		plt.yscale('log')
		plt.xscale('log')
		plt.show()

	return G,G0

###############################
# Input parameter values here #
###############################

c = 1000.
x = 1.
M = 10
ku = 1.
kb = 10.
N = 100
P = 500
w_lo = 1.e-3
w_hi = 1.e3
w_n = 50
omega = np.transpose(np.logspace(np.log10(w_lo),np.log10(w_hi),w_n))

output = rheo_calc(c,x,M,ku,kb,N,P,omega,plot_calc=True,plot_calc_all=True)