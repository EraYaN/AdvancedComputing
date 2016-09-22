# You need https://www.powershellgallery.com/packages/Invoke-MsBuild
$measured = Measure-Command {$buildSucceeded = Invoke-MsBuild -Path "MSBuild.csproj" -MsBuildParameters '/m /t:Clean;Rebuild' -ShowBuildOutputInCurrentWindow}

if ($buildSucceeded)
{ Write-Host "Build completed successfully in $measured" }
else
{ Write-Host "Build failed after $measured. Check the build log file for errors." }