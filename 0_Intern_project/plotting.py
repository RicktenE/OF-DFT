def plotting_normal(u,title):
    rplot = (mesh.coordinates())
    x = rplot
    y = [u(v) for v in rplot]
    
    plt.figure()
    plt.title(title)
    plt.xlabel("Radial coordinate")
    plt.ylabel(title)
    plt.grid()
    plt.plot(x,y)
    plt.show()
    return 


def plotting_sqrt(u,title):
    rplot = (mesh.coordinates())
    x = np.sqrt(rplot)
    y = [v*sqrt(u(v)) for v in rplot] 
    plt.figure()
    plt.title(title)
    plt.xlabel("Radial coordinate")
    plt.ylabel(title)
    plt.grid()
    plt.plot(x,y)
    plt.show()
    return 