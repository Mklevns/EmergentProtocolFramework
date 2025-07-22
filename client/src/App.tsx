import { Switch, Route } from "wouter";
import { queryClient } from "./lib/queryClient";
import { QueryClientProvider } from "@tanstack/react-query";
import { Toaster } from "@/components/ui/toaster";
import { TooltipProvider } from "@/components/ui/tooltip";
import { Navigation } from "@/components/Navigation";
import NotFound from "@/pages/not-found";
import Dashboard from "@/pages/dashboard";
import Training from "@/pages/training";
import Visualization from "@/pages/visualization";
import AdvancedLearning from "@/pages/AdvancedLearning";
import ResearchDashboard from "@/pages/ResearchDashboard";
import Config from "@/pages/config";

function Router() {
  return (
    <Switch>
      <Route path="/" component={Dashboard} />
      <Route path="/dashboard" component={Dashboard} />
      <Route path="/training" component={Training} />
      <Route path="/visualization" component={Visualization} />
      <Route path="/advanced-learning" component={AdvancedLearning} />
      <Route path="/research" component={ResearchDashboard} />
      <Route path="/config" component={Config} />
      <Route component={NotFound} />
    </Switch>
  );
}

function App() {
  return (
    <QueryClientProvider client={queryClient}>
      <TooltipProvider>
        <Navigation />
        <Toaster />
        <Router />
      </TooltipProvider>
    </QueryClientProvider>
  );
}

export default App;
